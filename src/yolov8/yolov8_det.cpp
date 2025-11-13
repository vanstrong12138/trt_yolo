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

// 原有函数不变
void serialize_engine(std::string& wts_name, std::string& engine_name, int& is_p, std::string& sub_type, float& gd,
                      float& gw, int& max_channels) {
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();
    IHostMemory* serialized_engine = nullptr;

    if (is_p == 6) {
        serialized_engine = buildEngineYolov8DetP6(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels);
    } else if (is_p == 2) {
        serialized_engine = buildEngineYolov8DetP2(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels);
    } else {
        serialized_engine = buildEngineYolov8Det(builder, config, DataType::kFLOAT, wts_name, gd, gw, max_channels);
    }

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
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
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
    // Create GPU buffers on device
    CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
    if (cuda_post_process == "c") {
        *output_buffer_host = new float[kBatchSize * kOutputSize];
    } else if (cuda_post_process == "g") {
        if (kBatchSize > 1) {
            std::cerr << "Do not yet support GPU post processing for multiple batches" << std::endl;
            exit(0);
        }
        // Allocate memory for decode_ptr_host and copy to device
        *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
        CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
    }
}

void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, int batchsize,
           float* decode_ptr_host, float* decode_ptr_device, int model_bboxes, std::string cuda_post_process) {
    // infer on the batch asynchronously, and DMA output back to host
    auto start = std::chrono::system_clock::now();
    context.setInputTensorAddress(kInputTensorName, buffers[0]);
    context.setOutputTensorAddress(kOutputTensorName, buffers[1]);
    context.enqueueV3(stream);
    if (cuda_post_process == "c") {
        CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
                  << "ms" << std::endl;
    } else if (cuda_post_process == "g") {
        CUDA_CHECK(
                cudaMemsetAsync(decode_ptr_device, 0, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
        cuda_decode((float*)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
        cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);  //cuda nms
        CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                   sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), cudaMemcpyDeviceToHost,
                                   stream));
        auto end = std::chrono::system_clock::now();
        std::cout << "inference and gpu postprocess time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
}

// ROS2 YOLOv8节点类
class YOLOv8ROSNode : public rclcpp::Node {
public:
    YOLOv8ROSNode() : Node("yolov8_node") {
        // 从参数服务器获取参数
        this->declare_parameter<std::string>("mode", "inference");  // "serialize" 或 "inference"
        this->declare_parameter<std::string>("wts_path", "");
        this->declare_parameter<std::string>("engine_path", "");
        this->declare_parameter<std::string>("model_type", "n");  // 这行应该是string // n/s/m/l/x
        this->declare_parameter<std::string>("cuda_post_process", "c");
        this->declare_parameter<std::string>("image_topic", "/camera/image_raw");
        this->declare_parameter<std::string>("output_topic", "/yolov8/detections");
        this->declare_parameter<std::string>("labels_file", "../coco.txt");
        this->declare_parameter<bool>("publish_image", true);
        
        initialize();
    }

    ~YOLOv8ROSNode() {
        cleanup();
    }

private:
    void initialize() {
        RCLCPP_INFO(this->get_logger(), "Initializing YOLOv8 ROS2 Node...");
        
        // 获取参数
        std::string mode = this->get_parameter("mode").as_string();
        std::string wts_path = this->get_parameter("wts_path").as_string();
        std::string engine_path = this->get_parameter("engine_path").as_string();
        std::string model_type = this->get_parameter("model_type").as_string();
        
        if (mode == "serialize") {
            // 序列化模式
            serializeModel(wts_path, engine_path, model_type);
        } else if (mode == "inference") {
            // 推理模式
            initializeInference(engine_path);
        } else {
            RCLCPP_ERROR(this->get_logger(), "Invalid mode: %s. Use 'serialize' or 'inference'.", mode.c_str());
        }
    }
    
    void serializeModel(const std::string& wts_path, const std::string& engine_path, const std::string& model_type) {
        RCLCPP_INFO(this->get_logger(), "Starting model serialization...");
        
        if (wts_path.empty() || engine_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "WTS path or engine path is empty for serialization mode!");
            return;
        }
        
        int is_p = 0;
        float gd = 0.0f, gw = 0.0f;
        int max_channels = 0;
        
        // 解析模型类型参数
        if (model_type[0] == 'n') {
            gd = 0.33; gw = 0.25; max_channels = 1024;
        } else if (model_type[0] == 's') {
            gd = 0.33; gw = 0.50; max_channels = 1024;
        } else if (model_type[0] == 'm') {
            gd = 0.67; gw = 0.75; max_channels = 576;
        } else if (model_type[0] == 'l') {
            gd = 1.0; gw = 1.0; max_channels = 512;
        } else if (model_type[0] == 'x') {
            gd = 1.0; gw = 1.25; max_channels = 640;
        } else {
            RCLCPP_ERROR(this->get_logger(), "Invalid model type: %s", model_type.c_str());
            return;
        }
        
        if (model_type.size() == 2) {
            if (model_type[1] == '6') {
                is_p = 6;
            } else if (model_type[1] == '2') {
                is_p = 2;
            }
        }
        
        std::string wts_name = wts_path;
        std::string engine_name = engine_path;
        std::string sub_type = model_type;
        
        serialize_engine(wts_name, engine_name, is_p, sub_type, gd, gw, max_channels);
        RCLCPP_INFO(this->get_logger(), "Engine serialized successfully to: %s", engine_path.c_str());
        
        // 序列化完成后关闭节点
        rclcpp::shutdown();
    }
    
    void initializeInference(std::string& engine_path) {
        RCLCPP_INFO(this->get_logger(), "Starting inference mode initialization...");
        
        if (engine_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Engine path parameter is empty for inference mode!");
            return;
        }
        
        // 获取推理相关参数
        cuda_post_process_ = this->get_parameter("cuda_post_process").as_string();
        std::string image_topic = this->get_parameter("image_topic").as_string();
        std::string labels_file = this->get_parameter("labels_file").as_string();
        bool publish_image = this->get_parameter("publish_image").as_bool();
        
        // 设置CUDA设备
        cudaSetDevice(kGpuId);
        
        // 反序列化引擎
        deserialize_engine(engine_path, &runtime_, &engine_, &context_);
        
        // 创建CUDA流
        CUDA_CHECK(cudaStreamCreate(&stream_));
        cuda_preprocess_init(kMaxInputImageSize);
        
        // 获取模型输出维度
        const char* output_name = engine_ -> getIOTensorName(1);

        auto out_dims = engine_->getTensorShape(output_name);

        model_bboxes_ = out_dims.d[0];
        
        // 准备缓冲区
        prepare_buffer(engine_, &device_buffers_[0], &device_buffers_[1], &output_buffer_host_, 
                      &decode_ptr_host_, &decode_ptr_device_, cuda_post_process_);

        read_labels(labels_file, labels_map_);
        assert(kNumClass == labels_map_.size());
        
        // 创建图像订阅者
        image_sub_ = image_transport::create_subscription(
            this,
            image_topic,
            std::bind(&YOLOv8ROSNode::imageCallback, this, std::placeholders::_1),
            "raw",
            rmw_qos_profile_sensor_data
        );
        
        // 创建检测结果发布者
        detection_pub_ = this->create_publisher<vision_msgs::msg::Detection2DArray>("/yolov8/detections", 10);
        
        if (publish_image) {
            image_pub_ = image_transport::create_publisher(this, "/yolov8/detection_image");
        }
        
        RCLCPP_INFO(this->get_logger(), "YOLOv8 ROS2 Node inference mode initialized successfully");
    }
    
    void cleanup() {
        RCLCPP_INFO(this->get_logger(), "Cleaning up YOLOv8 ROS2 Node...");
        
        // 只清理推理模式相关的资源
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        if (device_buffers_[0]) {
            CUDA_CHECK(cudaFree(device_buffers_[0]));
        }
        if (device_buffers_[1]) {
            CUDA_CHECK(cudaFree(device_buffers_[1]));
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
        
        // 销毁引擎
        if (context_) {
            delete context_;
        }
        if (engine_) {
            delete engine_;
        }
        if (runtime_) {
            delete runtime_;
        }
        
        RCLCPP_INFO(this->get_logger(), "YOLOv8 ROS2 Node cleanup completed");
    }
    
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        try {
            // 转换图像消息为OpenCV格式
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
            cv::Mat img = cv_ptr->image;
            
            // 单批次处理（适应实时需求）
            std::vector<cv::Mat> img_batch = {img};
            
            // 预处理
            cuda_batch_preprocess(img_batch, device_buffers_[0], kInputW, kInputH, stream_);
            
            // 推理
            infer(*context_, stream_, (void**)device_buffers_, output_buffer_host_, 1, 
                  decode_ptr_host_, decode_ptr_device_, model_bboxes_, cuda_post_process_);
            
            // 后处理
            std::vector<std::vector<Detection>> res_batch;
            if (cuda_post_process_ == "c") {
                batch_nms(res_batch, output_buffer_host_, img_batch.size(), kOutputSize, kConfThresh, kNmsThresh);
            } else if (cuda_post_process_ == "g") {
                batch_process(res_batch, decode_ptr_host_, img_batch.size(), bbox_element, img_batch);
            }
            
            // 发布检测结果
            publishDetections(res_batch[0], msg->header);
            
            // 绘制边界框并发布图像
            if (image_pub_.getNumSubscribers() > 0) {
                draw_bbox_with_label(img_batch, res_batch, labels_map_);
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
            
            // 添加置信度
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
    
    // YOLOv8推理相关成员变量
    IRuntime* runtime_ = nullptr;
    ICudaEngine* engine_ = nullptr;
    IExecutionContext* context_ = nullptr;
    cudaStream_t stream_ = nullptr;
    float* device_buffers_[2] = {nullptr, nullptr};
    float* output_buffer_host_ = nullptr;
    float* decode_ptr_host_ = nullptr;
    float* decode_ptr_device_ = nullptr;
    int model_bboxes_ = 0;
    std::string cuda_post_process_;
    std::unordered_map<int, std::string> labels_map_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<YOLOv8ROSNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    
    return 0;
}