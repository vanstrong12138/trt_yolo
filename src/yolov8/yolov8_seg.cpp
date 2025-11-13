#include <rclcpp/rclcpp.hpp>
#include <image_transport/image_transport.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"
#include <cv_bridge/cv_bridge.h>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <sensor_msgs/msg/image.hpp>

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * (sizeof(Detection) - sizeof(float) * 51) / sizeof(float) + 1;
const static int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);

static cv::Rect get_downscale_rect(float bbox[4], float scale) {
    float left = bbox[0];
    float top = bbox[1];
    float right = bbox[0] + bbox[2];
    float bottom = bbox[1] + bbox[3];

    left = left < 0 ? 0 : left;
    top = top < 0 ? 0 : top;
    right = right > 640 ? 640 : right;
    bottom = bottom > 640 ? 640 : bottom;

    left /= scale;
    top /= scale;
    right /= scale;
    bottom /= scale;
    return cv::Rect(int(left), int(top), int(right - left), int(bottom - top));
}

std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets) {
    std::vector<cv::Mat> masks;
    for (size_t i = 0; i < dets.size(); i++) {
        cv::Mat mask_mat = cv::Mat::zeros(kInputH / 4, kInputW / 4, CV_32FC1);
        auto r = get_downscale_rect(dets[i].bbox, 4);

        for (int x = r.x; x < r.x + r.width; x++) {
            for (int y = r.y; y < r.y + r.height; y++) {
                float e = 0.0f;
                for (int j = 0; j < 32; j++) {
                    e += dets[i].mask[j] * proto[j * proto_size / 32 + y * mask_mat.cols + x];
                }
                e = 1.0f / (1.0f + expf(-e));
                mask_mat.at<float>(y, x) = e;
            }
        }
        cv::resize(mask_mat, mask_mat, cv::Size(kInputW, kInputH));
        masks.push_back(mask_mat);
    }
    return masks;
}

void serialize_engine(std::string& wts_name, std::string& engine_name, std::string& sub_type, float& gd, float& gw,
                      int& max_channels) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* serialized_engine = nullptr;

    serialized_engine = buildEngineYolov8Seg(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels);

    assert(serialized_engine);
    std::ofstream p(engine_name, std::ios::binary);
    if (!p) {
        std::cout << "could not open plan output file" << std::endl;
        assert(false);
    }
    p.write(reinterpret_cast<const char*>(serialized_engine->data()), serialized_engine->size());

    delete serialized_engine;
    delete config;
    delete builder;
}

void deserialize_engine(std::string& engine_name, IRuntime** runtime, ICudaEngine** engine,
                        IExecutionContext** context) {
    std::ifstream file(engine_name, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << engine_name << " error!" << std::endl;
        assert(false);
    }
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    char* serialized_engine = new char[size];
    assert(serialized_engine);
    file.read(serialized_engine, size);
    file.close();

    *runtime = createInferRuntime(gLogger);
    assert(*runtime);
    *engine = (*runtime)->deserializeCudaEngine(serialized_engine, size);
    assert(*engine);
    *context = (*engine)->createExecutionContext();
    assert(*context);
    delete[] serialized_engine;
}

void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                    float** output_seg_buffer_device, float** output_buffer_host, float** output_seg_buffer_host,
                    float** decode_ptr_host, float** decode_ptr_device, std::string cuda_post_process) {
    assert(engine->getNbIOTensors() == 3);
    TensorIOMode input_mode = engine->getTensorIOMode(kInputTensorName);
    if (input_mode != TensorIOMode::kINPUT) {
        std::cerr << kInputTensorName << " should be input tensor" << std::endl;
        assert(false);
    }
    TensorIOMode output_mode = engine->getTensorIOMode(kOutputTensorName);
    if (output_mode != TensorIOMode::kOUTPUT) {
        std::cerr << kOutputTensorName << " should be output tensor" << std::endl;
        assert(false);
    }
    TensorIOMode proto_mode = engine->getTensorIOMode(kProtoTensorName);
    if (proto_mode != TensorIOMode::kOUTPUT) {
        std::cerr << kProtoTensorName << " should be output tensor" << std::endl;
        assert(false);
    }
    
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_seg_buffer_device, kBatchSize * kOutputSegSize * sizeof(float)));

    if (cuda_post_process == "c") {
        *output_buffer_host = new float[kBatchSize * kOutputSize];
        *output_seg_buffer_host = new float[kBatchSize * kOutputSegSize];
    } else if (cuda_post_process == "g") {
        if (kBatchSize > 1) {
            std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
            exit(0);
        }
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, float* output_seg,
           int batchsize, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes,
           std::string cuda_post_process) {
    auto start = std::chrono::system_clock::now();
    context.setInputTensorAddress(kInputTensorName, buffers[0]);
    context.setOutputTensorAddress(kOutputTensorName, buffers[1]);
    context.setOutputTensorAddress(kProtoTensorName, buffers[2]);
    context.enqueueV3(stream);
    
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(output_seg, buffers[2], batchsize * kOutputSegSize * sizeof(float), cudaMemcpyDeviceToHost, stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float*)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost, stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference and gpu postprocess time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ROS2 YOLOv8分割节点类
class YOLOv8SegROSNode : public rclcpp::Node {
public:
    YOLOv8SegROSNode() : Node("yolov8_seg_node") {
        this->declare_parameter<std::string>("mode", "inference");
        this->declare_parameter<std::string>("wts_path", "");
        this->declare_parameter<std::string>("engine_path", "");
        this->declare_parameter<std::string>("model_type", "n");
        this->declare_parameter<std::string>("cuda_post_process", "c");
        this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
        this->declare_parameter<std::string>("output_topic", "/yolov8/segmentation");
        this->declare_parameter<std::string>("labels_file", "../coco.txt");
        this->declare_parameter<bool>("publish_image", true);
        
        initialize();
    }

    ~YOLOv8SegROSNode() {
        cleanup();
    }

private:
    void initialize() {
        RCLCPP_INFO(this->get_logger(), "Initializing YOLOv8 Segmentation ROS2 Node...");
        
        std::string mode = this->get_parameter("mode").as_string();
        std::string wts_path = this->get_parameter("wts_path").as_string();
        std::string engine_path = this->get_parameter("engine_path").as_string();
        std::string model_type = this->get_parameter("model_type").as_string();
        
        if (mode == "serialize") {
            serializeModel(wts_path, engine_path, model_type);
        } else if (mode == "inference") {
            initializeInference(engine_path);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Invalid mode: %s. Use 'serialize' or 'inference'.", mode.c_str());
        }
    }
    
    void serializeModel(const std::string& wts_path, const std::string& engine_path, const std::string& model_type) {
        RCLCPP_INFO(this->get_logger(), "Starting segmentation model serialization...");
        
        if (wts_path.empty() || engine_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "WTS path or engine path is empty for serialization mode!");
            return;
        }
        
        float gd = 0.0f, gw = 0.0f;
        int max_channels = 0;
        
        if (model_type == "n") {
            gd = 0.33; gw = 0.25; max_channels = 1024;
        } else if (model_type == "s") {
            gd = 0.33; gw = 0.50; max_channels = 1024;
        } else if (model_type == "m") {
            gd = 0.67; gw = 0.75; max_channels = 576;
        } else if (model_type == "l") {
            gd = 1.0; gw = 1.0; max_channels = 512;
        } else if (model_type == "x") {
            gd = 1.0; gw = 1.25; max_channels = 640;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Invalid model type: %s", model_type.c_str());
            return;
        }
        
        std::string wts_name = wts_path;
        std::string engine_name = engine_path;
        std::string sub_type = model_type;
        
        serialize_engine(wts_name, engine_name, sub_type, gd, gw, max_channels);
        RCLCPP_INFO(this->get_logger(), "Segmentation engine serialized successfully to: %s", engine_path.c_str());
        
        rclcpp::shutdown();
    }
    
    void initializeInference(std::string& engine_path) {
        RCLCPP_INFO(this->get_logger(), "Starting segmentation inference mode initialization...");
        
        if (engine_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Engine path parameter is empty for inference mode!");
            return;
        }
        
        cuda_post_process_ = this->get_parameter("cuda_post_process").as_string();
        std::string image_topic = this->get_parameter("image_topic").as_string();
        std::string labels_file = this->get_parameter("labels_file").as_string();
        bool publish_image = this->get_parameter("publish_image").as_bool();
        
        if (cuda_post_process_ == "g") {
            RCLCPP_WARN(this->get_logger(), "GPU segmentation postprocess is not supported yet, falling back to CPU");
            cuda_post_process_ = "c";
        }
        
        cudaSetDevice(kGpuId);
        
        deserialize_engine(engine_path, &runtime_, &engine_, &context_);
        
        CUDA_CHECK(cudaStreamCreate(&stream_));
        cuda_preprocess_init(kMaxInputImageSize);
        
        auto out_dims = engine_->getTensorShape(kOutputTensorName);
        model_bboxes_ = out_dims.d[1];
        
        prepare_buffer(engine_, &device_buffers_[0], &device_buffers_[1], &device_buffers_[2], 
                      &output_buffer_host_, &output_seg_buffer_host_, &decode_ptr_host_, 
                      &decode_ptr_device_, cuda_post_process_);
        
        read_labels(labels_file, labels_map_);
        assert(kNumClass == labels_map_.size());
        
        image_sub_ = image_transport::create_subscription(
            this,
            image_topic,
            std::bind(&YOLOv8SegROSNode::imageCallback, this, std::placeholders::_1),
            "raw",
            rmw_qos_profile_sensor_data
        );
        
        detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/yolov8/segmentation", 10);
        
        if (publish_image) {
            image_pub_ = image_transport::create_publisher(this, "/yolov8/segmentation_image");
        }
        
        RCLCPP_INFO(this->get_logger(), "YOLOv8 Segmentation ROS2 Node initialized successfully");
    }
    
    void cleanup() {
        RCLCPP_INFO(this->get_logger(), "Cleaning up YOLOv8 Segmentation ROS2 Node...");
        
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        if (device_buffers_[0]) {
            CUDA_CHECK(cudaFree(device_buffers_[0]));
        }
        if (device_buffers_[1]) {
            CUDA_CHECK(cudaFree(device_buffers_[1]));
        }
        if (device_buffers_[2]) {
            CUDA_CHECK(cudaFree(device_buffers_[2]));
        }
        if (decode_ptr_device_) {
            CUDA_CHECK(cudaFree(decode_ptr_device_));
        }
        if (decode_ptr_host_) {
            delete[] decode_ptr_host_;
        }
        if (output_buffer_host_) {
            delete[] output_buffer_host_;
        }
        if (output_seg_buffer_host_) {
            delete[] output_seg_buffer_host_;
        }
        
        cuda_preprocess_destroy();
        
        if (context_) {
            delete context_;
        }
        if (engine_) {
            delete engine_;
        }
        if (runtime_) {
            delete runtime_;
        }
        
        RCLCPP_INFO(this->get_logger(), "YOLOv8 Segmentation ROS2 Node cleanup completed");
    }
    
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            cv::Mat img = cv_ptr->image;
            
            std::vector<cv::Mat> img_batch = {img};
            
            cuda_batch_preprocess(img_batch, device_buffers_[0], kInputW, kInputH, stream_);
            
            infer(*context_, stream_, (void**)device_buffers_, output_buffer_host_, output_seg_buffer_host_, 1, 
                  decode_ptr_host_, decode_ptr_device_, model_bboxes_, cuda_post_process_);
            
            std::vector<std::vector<Detection>> res_batch;
            if (cuda_post_process_ == "c") {
                batch_nms(res_batch, output_buffer_host_, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
                
                for (size_t b = 0; b < img_batch.size(); b++) {
                    auto& res = res_batch[b];
                    cv::Mat img = img_batch[b];
                    auto masks = process_mask(&output_seg_buffer_host_[b * kOutputSegSize], kOutputSegSize, res);
                    
                    if (image_pub_.getNumSubscribers() > 0) {
                        draw_mask_bbox(img, res, masks, labels_map_);
                        publishImage(img, msg->header);
                    }
                    
                    publishDetections(res, masks, msg->header);
                }
            } else if (cuda_post_process_ == "g") {
                RCLCPP_WARN(this->get_logger(), "GPU segmentation postprocess not supported");
            }
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge error: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Image processing error: %s", e.what());
        }
    }
    
    void publishDetections(const std::vector<Detection>& detections, const std::vector<cv::Mat>& masks, const std_msgs::msg::Header& header) {
        auto detection_msg = vision_msgs::msg::Detection2DArray();
        detection_msg.header = header;
        
        for (size_t i = 0; i < detections.size(); i++) {
            const auto& det = detections[i];
            if (det.conf < kConfThresh) continue;
            
            vision_msgs::msg::Detection2D detection;
            detection.bbox.center.position.x = det.bbox[0];
            detection.bbox.center.position.y = det.bbox[1];
            detection.bbox.size_x = det.bbox[2];
            detection.bbox.size_y = det.bbox[3];
            detection.id = std::to_string(det.class_id);
            
            vision_msgs::msg::ObjectHypothesisWithPose hypothesis;
            hypothesis.hypothesis.class_id = std::to_string(det.class_id);
            hypothesis.hypothesis.score = det.conf;
            detection.results.push_back(hypothesis);
            
            detection_msg.detections.push_back(detection);
        }
        
        detection_pub_->publish(detection_msg);
    }
    
    void publishImage(const cv::Mat& img, const std_msgs::msg::Header& header) {
        cv_bridge::CvImage cv_image;
        cv_image.header = header;
        cv_image.encoding = "rgb8";
        cv_image.image = img;
        
        image_pub_.publish(cv_image.toImageMsg());
    }

    // ROS2相关成员变量
    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;
    
    // YOLOv8分割推理相关成员变量
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    float* device_buffers_[3] = {nullptr, nullptr, nullptr};
    float* output_buffer_host_ = nullptr;
    float* output_seg_buffer_host_ = nullptr;
    float* decode_ptr_host_ = nullptr;
    float* decode_ptr_device_ = nullptr;
    int model_bboxes_ = 0;
    std::string cuda_post_process_;
    std::unordered_map<int, std::string> labels_map_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YOLOv8SegROSNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}