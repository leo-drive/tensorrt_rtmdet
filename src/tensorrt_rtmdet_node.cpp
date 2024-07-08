//
// Created by bzeren on 14.06.2024.
//

#include "tensorrt_rtmdet/tensorrt_rtmdet_node.hpp"

#include <fstream>
#include <dlfcn.h>

namespace tensorrt_rtmdet {
    TrtRTMDetNode::TrtRTMDetNode(const rclcpp::NodeOptions &node_options)
            : Node("tensorrt_rtmdet", node_options) {

        std::string model_path = declare_parameter<std::string>("model_path");
        std::string color_map_path = declare_parameter<std::string>("color_map_path");
        std::string precision = declare_parameter<std::string>("precision");
        std::vector<double> mean = declare_parameter<std::vector<double>>("mean");
        std::vector<double> std = declare_parameter<std::vector<double>>("std");
        int number_classes = declare_parameter<int>("number_classes");
        double score_threshold = declare_parameter<double>("score_threshold");
        double nms_threshold = declare_parameter<double>("nms_threshold");
        double mask_threshold = declare_parameter<double>("mask_threshold");
        std::string calibration_algorithm = declare_parameter<std::string>("calibration_algorithm");
        int dla_core_id = declare_parameter<int>("dla_core_id");
        bool quantize_first_layer = declare_parameter<bool>("quantize_first_layer");
        bool quantize_last_layer = declare_parameter<bool>("quantize_last_layer");
        bool profile_per_layer = declare_parameter<bool>("profile_per_layer");
        double clip_value = declare_parameter<double>("clip_value");
        bool preprocess_on_gpu = declare_parameter<bool>("preprocess_on_gpu");
        std::string calibration_image_list_path = declare_parameter<std::string>("calibration_image_list_path");
        std::vector<std::string> plugin_paths = declare_parameter<std::vector<std::string>>("plugin_paths");

        tensorrt_common::BuildConfig build_config(
                calibration_algorithm, dla_core_id, quantize_first_layer, quantize_last_layer, profile_per_layer,
                clip_value
        );

        const double norm_factor = 1.0;
        const std::string cache_dir = "";
        const tensorrt_common::BatchConfig batch_config{1, 1, 1};
        const size_t max_workspace_size = (1 << 30);

        mean_ = std::vector<float>(mean.begin(), mean.end());
        std_ = std::vector<float>(std.begin(), std.end());

        trt_rtmdet_ = std::make_unique<tensorrt_rtmdet::TrtRTMDet>(
                model_path, precision, number_classes, score_threshold, nms_threshold, mask_threshold, build_config,
                preprocess_on_gpu, calibration_image_list_path, norm_factor, mean_, std_, cache_dir, batch_config,
                max_workspace_size, color_map_path, plugin_paths
        );

        if (declare_parameter("build_only", false)) {
            RCLCPP_INFO(this->get_logger(), "TensorRT engine file is built and exit.");
            rclcpp::shutdown();
        }

        cv::VideoCapture cap("/home/bzeren/projects/labs/rtmdet/road.mp4");

        while (rclcpp::ok()) {
            cv::Mat frame;
            // Capture frame-by-frame
            cap >> frame;

            if (frame.empty())
                break;

            tensorrt_rtmdet::ObjectArrays objects;

            std::cout << "Start inference" << std::endl;
            if (!trt_rtmdet_->doInference({frame}, objects)) {
                RCLCPP_WARN(this->get_logger(), "Fail to inference");
                return;
            }

            for (const auto &object : objects) {
                for (const auto &obj : object) {
                    std::cout << "|| Class: " << obj.class_id << " Score: " << obj.score << std::endl;
                }
            }

            std::cout << "End inference" << std::endl;
        }

    }
} // namespace tensorrt_rtmdet

#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(tensorrt_rtmdet::TrtRTMDetNode)