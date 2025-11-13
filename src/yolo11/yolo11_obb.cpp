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

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;

void serialize_engine(std::string& wts_name, std::string& engine_name, std::string& type, float& gd, float& gw,
                      int& max_channels) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* serialized_engine = nullptr;

    serialized_engine = buildEngineYolo11Obb(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels, type);

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
                    float** output_buffer_host, float** decode_ptr_host, float** decode_ptr_device,
                    std::string cuda_post_process) {
    assert(engine->getNbIOTensors() == 2);

    nvinfer1::Dims input_dims = engine->getTensorShape(kInputTensorName);
    nvinfer1::Dims output_dims = engine->getTensorShape(kOutputTensorName);

    int input_size = kBatchSize * 3 * kInputH * kInputW * sizeof(float);
    int output_size = kBatchSize * kOutputSize * sizeof(float);

    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kObbInputH * kObbInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));

    if (cuda_post_process == "c") {
        *output_buffer_host = new float[kBatchSize * kOutputSize];
    } else if (cuda_post_process == "g") {
        if (kBatchSize > 1) {
            std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
            exit(0);
        }
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void infer(IExecutionContext& context, cudaStream_t& stream, float* input_buffer_device, float* output_buffer_device, 
           float* output, int batchsize, float* decode_ptr_host, float* decode_ptr_device, 
           int model_bboxes, std::string cuda_post_process) {
    
    // 修复：设置输入输出张量地址
    context.setTensorAddress(kInputTensorName, input_buffer_device);
    context.setTensorAddress(kOutputTensorName, output_buffer_device);
    
    auto start = std::chrono::system_clock::now();
    context.enqueueV3(stream);
    
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output, output_buffer_device, batchsize * kOutputSize * sizeof(float), 
                                  cudaMemcpyDeviceToHost, stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode_obb((float*)output_buffer_device, model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms_obb(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                  sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), 
                                  cudaMemcpyDeviceToHost, stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference and gpu postprocess time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

class YOLOv11ObbROSNode : public rclcpp::Node {
public:
    YOLOv11ObbROSNode() : Node("yolov11_obb_node") {
        this->declare_parameter<std::string>("mode", "inference");
        this->declare_parameter<std::string>("wts_path", "");
        this->declare_parameter<std::string>("engine_path", "");
        this->declare_parameter<std::string>("model_type", "n");
        this->declare_parameter<std::string>("cuda_post_process", "c");
        this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
        this->declare_parameter<std::string>("output_topic", "/yolov11_obb/detections");
        this->declare_parameter<bool>("publish_image", true);
        
        initialize();
    }

    ~YOLOv11ObbROSNode() {
        cleanup();
    }

private:
    void initialize() {
        RCLCPP_INFO(this->get_logger(), "Initializing YOLOv11 OBB ROS2 Node...");
        
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
        RCLCPP_INFO(this->get_logger(), "Starting OBB model serialization...");
        
        if (wts_path.empty() || engine_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "WTS path or engine path is empty for serialization mode!");
            return;
        }
        
        float gd = 0, gw = 0;
        int max_channels = 0;
        std::string type;
        
        if (model_type[0] == 'n') {
            gd = 0.50;
            gw = 0.25;
            max_channels = 1024;
            type = "n";
        } else if (model_type[0] == 's') {
            gd = 0.50;
            gw = 0.50;
            max_channels = 1024;
            type = "s";
        } else if (model_type[0] == 'm') {
            gd = 0.50;
            gw = 1.00;
            max_channels = 512;
            type = "m";
        } else if (model_type[0] == 'l') {
            gd = 1.0;
            gw = 1.0;
            max_channels = 512;
            type = "l";
        } else if (model_type[0] == 'x') {
            gd = 1.0;
            gw = 1.50;
            max_channels = 512;
            type = "x";
        } else {
            RCLCPP_ERROR(this->get_logger(), "Invalid model type: %s", model_type.c_str());
            return;
        }
        
        std::string wts_name = wts_path;
        std::string engine_name = engine_path;
        
        serialize_engine(wts_name, engine_name, type, gd, gw, max_channels);
        RCLCPP_INFO(this->get_logger(), "OBB Engine serialized successfully to: %s", engine_path.c_str());
        
        rclcpp::shutdown();
    }
    
    void initializeInference(std::string& engine_path) {
        RCLCPP_INFO(this->get_logger(), "Starting OBB inference mode initialization...");
        
        if (engine_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Engine path parameter is empty for inference mode!");
            return;
        }
        
        cuda_post_process_ = this->get_parameter("cuda_post_process").as_string();
        std::string image_topic = this->get_parameter("image_topic").as_string();
        bool publish_image = this->get_parameter("publish_image").as_bool();
        
        cudaSetDevice(kGpuId);
        
        deserialize_engine(engine_path, &runtime_, &engine_, &context_);
        
        CUDA_CHECK(cudaStreamCreate(&stream_));
        cuda_preprocess_init(kMaxInputImageSize);
        
        auto out_dims = engine_->getTensorShape(kOutputTensorName);
        model_bboxes_ = out_dims.d[0];
        
        prepare_buffer(engine_, &input_buffer_device_, &output_buffer_device_, &output_buffer_host_, 
                      &decode_ptr_host_, &decode_ptr_device_, cuda_post_process_);
        
        image_sub_ = image_transport::create_subscription(
            this,
            image_topic,
            std::bind(&YOLOv11ObbROSNode::imageCallback, this, std::placeholders::_1),
            "raw",
            rmw_qos_profile_sensor_data
        );
        
        detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/yolov11_obb/detections", 10);
        
        if (publish_image) {
            image_pub_ = image_transport::create_publisher(this, "/yolov11_obb/detection_image");
        }
        
        RCLCPP_INFO(this->get_logger(), "YOLOv11 OBB ROS2 Node inference mode initialized successfully");
    }
    
    void cleanup() {
        RCLCPP_INFO(this->get_logger(), "Cleaning up YOLOv11 OBB ROS2 Node...");
        
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        if (input_buffer_device_) {
            CUDA_CHECK(cudaFree(input_buffer_device_));
        }
        if (output_buffer_device_) {
            CUDA_CHECK(cudaFree(output_buffer_device_));
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
        
        RCLCPP_INFO(this->get_logger(), "YOLOv11 OBB ROS2 Node cleanup completed");
    }
    
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            cv::Mat img = cv_ptr->image;
            
            std::vector<cv::Mat> img_batch = {img};
            
            cuda_batch_preprocess(img_batch, input_buffer_device_, kObbInputW, kObbInputH, stream_);
            
            infer(*context_, stream_, input_buffer_device_, output_buffer_device_, output_buffer_host_, 1, 
                  decode_ptr_host_, decode_ptr_device_, model_bboxes_, cuda_post_process_);
            
            std::vector<std::vector<Detection>> res_batch;
            if (cuda_post_process_ == "c") {
                batch_nms_obb(res_batch, output_buffer_host_, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
            } else if (cuda_post_process_ == "g") {
                RCLCPP_WARN(this->get_logger(), "OBB postprocess is not support in GPU right now");
                return;
            }
            
            publishDetections(res_batch[0], msg->header);
            
            if (image_pub_.getNumSubscribers() > 0) {
                draw_bbox_obb(img_batch, res_batch);
                publishImage(img_batch[0], msg->header);
            }
            
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "CV bridge error: %s", e.what());
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Image processing error: %s", e.what());
        }
    }
    
    void publishDetections(const std::vector<Detection>& detections, const std_msgs::msg::Header& header) {
        auto detection_msg = vision_msgs::msg::Detection2DArray();
        detection_msg.header = header;
        
        for (const auto& det : detections) {
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

    image_transport::Subscriber image_sub_;
    image_transport::Publisher image_pub_;
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr detection_pub_;
    
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    float* input_buffer_device_ = nullptr;
    float* output_buffer_device_ = nullptr;
    float* output_buffer_host_ = nullptr;
    float* decode_ptr_host_ = nullptr;
    float* decode_ptr_device_ = nullptr;
    int model_bboxes_ = 0;
    std::string cuda_post_process_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YOLOv11ObbROSNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}