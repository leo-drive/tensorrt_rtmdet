//
// Created by bzeren on 14.06.2024.
//

#ifndef TENSORRT_RTMDET_TENSORRT_RTMDET_HPP
#define TENSORRT_RTMDET_TENSORRT_RTMDET_HPP

#include "tensorrt_rtmdet/calibrator.hpp"
#include "tensorrt_rtmdet/preprocess.hpp"
#include "tensorrt_rtmdet/postprocess.hpp"
#include "tier4_perception_msgs/msg/detected_object_with_mask.hpp"
#include "tier4_perception_msgs/msg/detected_objects_with_mask.hpp"

#include <cuda_utils/cuda_unique_ptr.hpp>
#include <cuda_utils/stream_unique_ptr.hpp>
#include <opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>

#include <tensorrt_common/tensorrt_common.hpp>

#include <memory>
#include <string>
#include <vector>

namespace tensorrt_rtmdet {
    using cuda_utils::CudaUniquePtr;
    using cuda_utils::CudaUniquePtrHost;
    using cuda_utils::makeCudaStream;
    using cuda_utils::StreamUniquePtr;

    struct Object {
        int32_t x1;
        int32_t y1;
        int32_t x2;
        int32_t y2;
        int32_t class_id;
        int32_t mask_index;
        float score;
    };

    using ObjectArray = std::vector<Object>;
    using ObjectArrays = std::vector<ObjectArray>;

    typedef struct LabelColor {
        std::string class_name;
        cv::Vec3b color;
        uint8_t label_id;
    } LabelColor;

    typedef std::map<int, LabelColor> ColorMap;

    class TrtRTMDet {
    public:
        TrtRTMDet(const std::string &model_path, const std::string &precision, const int num_class = 80,
                  const float score_threshold = 0.3, const float nms_threshold = 0.7,
                  const float mask_threshold = 200.0,
                  const tensorrt_common::BuildConfig build_config = tensorrt_common::BuildConfig(),
                  const bool use_gpu_preprocess = false, std::string calibration_image_list_file = std::string(),
                  const double norm_factor = 1.0, const std::vector<float> mean = {103.53, 116.28, 123.675},
                  const std::vector<float> std = {57.375, 57.12, 58.395},
                  [[maybe_unused]] const std::string &cache_dir = "",
                  const tensorrt_common::BatchConfig &batch_config = {1, 1, 1},
                  const size_t max_workspace_size = (1 << 30), const std::string &color_map_path = "",
                  const std::vector<std::string> &plugin_paths = {});

        ~TrtRTMDet();

        bool doInference(const std::vector<cv::Mat> &images, ObjectArrays &objects,
                         tier4_perception_msgs::msg::DetectedObjectsWithMask &detected_objects_with_mask);

        bool doInferenceWithRoi(
                const std::vector<cv::Mat> &images, ObjectArrays &objects, const std::vector<cv::Rect> &roi,
                tier4_perception_msgs::msg::DetectedObjectsWithMask &detected_objects_with_mask);

        bool doMultiScaleInference(
                const cv::Mat &image, ObjectArrays &objects, const std::vector<cv::Rect> &roi);

        void printProfiling(void);


    private:
        void preprocess(const std::vector<cv::Mat> &images);

        void preprocessGpu(const std::vector<cv::Mat> &images);

        void preprocessWithRoi(const std::vector<cv::Mat> &images, const std::vector<cv::Rect> &rois);

        void preprocessWithRoiGpu(
                const std::vector<cv::Mat> &images, const std::vector<cv::Rect> &rois);

        void multiScalePreprocess(const cv::Mat &image, const std::vector<cv::Rect> &rois);

        void multiScalePreprocessGpu(const cv::Mat &image, const std::vector<cv::Rect> &rois);

        bool multiScaleFeedforward(const cv::Mat &image, int batch_size, ObjectArrays &objects);

        bool feedforward(const std::vector<cv::Mat> &images, ObjectArrays &objects,
                         tier4_perception_msgs::msg::DetectedObjectsWithMask &detected_objects_with_mask);

        void readColorMap(const std::string &color_map_path);

        void nmsSortedBboxes(const ObjectArray &input_objects, ObjectArray &output_objects) const;

        std::unique_ptr<tensorrt_common::TrtCommon> trt_common_;

        std::vector<float> input_h_;
        CudaUniquePtr<float[]> input_d_;
        CudaUniquePtr<float[]> out_dets_d_;
        CudaUniquePtr<int32_t[]> out_labels_d_;
        CudaUniquePtr<float[]> out_masks_d_;

        size_t out_elem_num_;
        size_t out_elem_num_per_batch_;
        CudaUniquePtr<float[]> out_prob_d_;

        StreamUniquePtr stream_{makeCudaStream()};

        int32_t max_detections_;
        float scale_width_;
        float scale_height_;

        // size of input image for model
        int model_input_width_;
        int model_input_height_;

        int num_class_;
        float score_threshold_;
        float nms_threshold_;
        float mask_threshold_;
        int batch_size_;
        CudaUniquePtrHost<float[]> out_prob_h_;

        // flag whether preprocess are performed on GPU
        bool use_gpu_preprocess_;
        // host buffer for preprocessing on GPU
        CudaUniquePtrHost<unsigned char[]> image_buf_h_;
        // device buffer for preprocessing on GPU
        CudaUniquePtr<unsigned char[]> image_buf_d_;
        // normalization factor used for preprocessing
        double norm_factor_;

        std::vector<int> output_strides_;

        int src_width_;
        int src_height_;

        std::vector<float> mean_;
        std::vector<float> std_;

        // host pointer for ROI
        CudaUniquePtrHost<Roi[]> roi_h_;
        // device pointer for ROI
        CudaUniquePtr<Roi[]> roi_d_;

        // Host pointer for outputs
        std::unique_ptr<float[]> out_dets_h_;
        std::unique_ptr<int32_t[]> out_labels_h_;
        std::unique_ptr<float[]> out_masks_h_;

        // Segmentation
        ColorMap color_map_;
    };
}

#endif //TENSORRT_RTMDET_TENSORRT_RTMDET_HPP
