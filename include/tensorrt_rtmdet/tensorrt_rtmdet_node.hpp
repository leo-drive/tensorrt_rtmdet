//
// Created by bzeren on 14.06.2024.
//

#ifndef TENSORRT_RTMDET__TENSORRT_RTMDET_NODE_HPP
#define TENSORRT_RTMDET__TENSORRT_RTMDET_NODE_HPP

#include "tensorrt_rtmdet/tensorrt_rtmdet.hpp"

#include <rclcpp/rclcpp.hpp>

#include <vector>

namespace tensorrt_rtmdet {
    class TrtRTMDetNode : public rclcpp::Node {
    public:
        explicit TrtRTMDetNode(const rclcpp::NodeOptions &node_options);

    private:
        std::unique_ptr<tensorrt_rtmdet::TrtRTMDet> trt_rtmdet_;

        std::vector<float> mean_;
        std::vector<float> std_;
    };
}

#endif // TENSORRT_RTMDET__TENSORRT_RTMDET_NODE_HPP
