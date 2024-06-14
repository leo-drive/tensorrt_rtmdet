//
// Created by bzeren on 14.06.2024.
//

#include "tensorrt_rtmdet/tensorrt_rtmdet.hpp"
#include "trt_batched_nms/batched_nms/trt_batched_nms.hpp"

#include <assert.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

static void trimLeft(std::string &s) {
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) { return !isspace(ch); }));
}

static void trimRight(std::string &s) {
    s.erase(find_if(s.rbegin(), s.rend(), [](int ch) { return !isspace(ch); }).base(), s.end());
}

std::string trim(std::string &s) {
    trimLeft(s);
    trimRight(s);
    return s;
}

bool fileExists(const std::string &file_name, bool verbose) {
    if (!std::experimental::filesystem::exists(std::experimental::filesystem::path(file_name))) {
        if (verbose) {
            std::cout << "File does not exist : " << file_name << std::endl;
        }
        return false;
    }
    return true;
}

std::vector<std::string> loadListFromTextFile(const std::string &filename) {
    assert(fileExists(filename, true));
    std::vector<std::string> list;

    std::ifstream f(filename);
    if (!f) {
        std::cout << "failed to open " << filename << std::endl;
        assert(0);
    }

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) {
            continue;
        } else {
            list.push_back(trim(line));
        }
    }

    return list;
}

std::vector<std::string> loadImageList(const std::string &filename, const std::string &prefix) {
    std::vector<std::string> fileList = loadListFromTextFile(filename);
    for (auto &file: fileList) {
        if (fileExists(file, false)) {
            continue;
        } else {
            std::string prefixed = prefix + file;
            if (fileExists(prefixed, false))
                file = prefixed;
            else
                std::cerr << "WARNING: couldn't find: " << prefixed << " while loading: " << filename
                          << std::endl;
        }
    }
    return fileList;
}

namespace tensorrt_rtmdet {
    TrtRTMDet::TrtRTMDet(const std::string &model_path, const std::string &precision,
                         [[maybe_unused]] const int num_class,
                         [[maybe_unused]] const float score_threshold, [[maybe_unused]] const float nms_threshold,
                         tensorrt_common::BuildConfig build_config,
                         [[maybe_unused]] const bool use_gpu_preprocess, std::string calibration_image_list_path,
                         const double norm_factor,
                         [[maybe_unused]] const std::string &cache_dir,
                         const tensorrt_common::BatchConfig &batch_config,
                         const size_t max_workspace_size, [[maybe_unused]] const std::string &color_map_path) {
        RCLCPP_INFO(rclcpp::get_logger("tensorrt_rtmdet"), "tensorrt_rtmdet has been started.");

        src_width_ = -1;
        src_height_ = -1;
        norm_factor_ = norm_factor;
        batch_size_ = batch_config[2];
        multitask_ = 0;

        if (precision == "int8") {
            if (build_config.clip_value <= 0.0) {
                if (calibration_image_list_path.empty()) {
                    throw std::runtime_error(
                            "calibration_image_list_path should be passed to generate int8 engine "
                            "or specify values larger than zero to clip_value.");
                }
            } else {
                // if clip value is larger than zero, calibration file is not needed
                calibration_image_list_path = "";
            }

            int max_batch_size = batch_size_;
            nvinfer1::Dims input_dims = tensorrt_common::get_input_dims(model_path);
            std::vector<std::string> calibration_images;
            if (calibration_image_list_path != "") {
                calibration_images = loadImageList(calibration_image_list_path, "");
            }
            tensorrt_rtmdet::ImageStream stream(max_batch_size, input_dims, calibration_images);
            fs::path calibration_table{model_path};
            std::string calibName = "";
            std::string ext = "";
            if (build_config.calib_type_str == "Entropy") {
                ext = "EntropyV2-";
            } else if (
                    build_config.calib_type_str == "Legacy" || build_config.calib_type_str == "Percentile") {
                ext = "Legacy-";
            } else {
                ext = "MinMax-";
            }

            ext += "calibration.table";
            calibration_table.replace_extension(ext);
            fs::path histogram_table{model_path};
            ext = "histogram.table";
            histogram_table.replace_extension(ext);

            std::unique_ptr<nvinfer1::IInt8Calibrator> calibrator;
            if (build_config.calib_type_str == "Entropy") {
                calibrator.reset(
                        new tensorrt_rtmdet::Int8EntropyCalibrator(stream, calibration_table, norm_factor_));

            } else if (
                    build_config.calib_type_str == "Legacy" || build_config.calib_type_str == "Percentile") {
                const double quantile = 0.999999;
                const double cutoff = 0.999999;
                calibrator.reset(new tensorrt_rtmdet::Int8LegacyCalibrator(
                        stream, calibration_table, histogram_table, norm_factor_, true, quantile, cutoff));
            } else {
                calibrator.reset(
                        new tensorrt_rtmdet::Int8MinMaxCalibrator(stream, calibration_table, norm_factor_));
            }

            trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
                    model_path, precision, std::move(calibrator), batch_config, max_workspace_size, build_config);
        } else {
            trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
                    model_path, precision, nullptr, batch_config, max_workspace_size, build_config);
        }
        trt_common_->setup();

        if (!trt_common_->isInitialized()) {
            return;
        }

        // TODO: read from config
        needs_output_decode_ = false;
        num_class_ = 80;
        score_threshold_ = 0.3;
        nms_threshold_ = 0.3;

        // GPU memory allocation
        const auto input_dims = trt_common_->getBindingDimensions(0);
        const auto input_size =
                std::accumulate(input_dims.d + 1, input_dims.d + input_dims.nbDims, 1, std::multiplies<int>());
        if (needs_output_decode_) {
            const auto output_dims = trt_common_->getBindingDimensions(1);
            input_d_ = cuda_utils::make_unique<float[]>(batch_config[2] * input_size);
            out_elem_num_ = std::accumulate(
                    output_dims.d + 1, output_dims.d + output_dims.nbDims, 1, std::multiplies<int>());
            out_elem_num_ = out_elem_num_ * batch_config[2];
            out_elem_num_per_batch_ = static_cast<int>(out_elem_num_ / batch_config[2]);
            out_prob_d_ = cuda_utils::make_unique<float[]>(out_elem_num_);
            out_prob_h_ = cuda_utils::make_unique_host<float[]>(out_elem_num_, cudaHostAllocPortable);
            int w = input_dims.d[3];
            int h = input_dims.d[2];
            int sum_tensors = (w / 8) * (h / 8) + (w / 16) * (h / 16) + (w / 32) * (h / 32);
            if (sum_tensors == output_dims.d[1]) {
                // 3head (8,16,32)
                output_strides_ = {8, 16, 32};
            } else {
                // 4head (8,16,32.4)
                // last is additional head for high resolution outputs
                output_strides_ = {8, 16, 32, 4};
            }
        } else {
            std::cout << "11111   " << input_size << std::endl;
            const auto out_scores_dims = trt_common_->getBindingDimensions(3);
            max_detections_ = out_scores_dims.d[1];
            input_d_ = cuda_utils::make_unique<float[]>(batch_config[2] * input_size);
            out_dets_d_ = cuda_utils::make_unique<float[]>(batch_config[2] * 500);
            out_labels_d_ = cuda_utils::make_unique<int32_t[]>(batch_config[2] * 100);
            out_masks_d_ = cuda_utils::make_unique<float[]>(batch_config[2] * 640 * 640 * 100);
        }
        if (multitask_) {
            // Allocate buffer for segmentation
            segmentation_out_elem_num_ = 0;
            for (int m = 0; m < multitask_; m++) {
                const auto output_dims =
                        trt_common_->getBindingDimensions(m + 2);  // 0 : input, 1 : output for detections
                size_t out_elem_num = std::accumulate(
                        output_dims.d + 1, output_dims.d + output_dims.nbDims, 1, std::multiplies<int>());
                out_elem_num = out_elem_num * batch_config[2];
                segmentation_out_elem_num_ += out_elem_num;
            }
            segmentation_out_elem_num_per_batch_ =
                    static_cast<int>(segmentation_out_elem_num_ / batch_config[2]);
            segmentation_out_prob_d_ = cuda_utils::make_unique<float[]>(segmentation_out_elem_num_);
            segmentation_out_prob_h_ =
                    cuda_utils::make_unique_host<float[]>(segmentation_out_elem_num_, cudaHostAllocPortable);
        }
        if (use_gpu_preprocess) {
            use_gpu_preprocess_ = true;
            image_buf_h_ = nullptr;
            image_buf_d_ = nullptr;
            if (multitask_) {
                argmax_buf_h_ = nullptr;
                argmax_buf_d_ = nullptr;
            }
        } else {
            use_gpu_preprocess_ = false;
        }
    }

    TrtRTMDet::~TrtRTMDet() {
        if (use_gpu_preprocess_) {
            if (image_buf_h_) {
                image_buf_h_.reset();
            }
            if (image_buf_d_) {
                image_buf_d_.reset();
            }
            if (argmax_buf_d_) {
                argmax_buf_d_.reset();
            }
        }
    }

    void TrtRTMDet::initPreprocessBuffer(int width, int height) {
        // if size of source input has been changed...
        if (src_width_ != -1 || src_height_ != -1) {
            if (width != src_width_ || height != src_height_) {
                // Free cuda memory to reallocate
                if (image_buf_h_) {
                    image_buf_h_.reset();
                }
                if (image_buf_d_) {
                    image_buf_d_.reset();
                }
            }
        }
        src_width_ = width;
        src_height_ = height;
        if (use_gpu_preprocess_) {
            auto input_dims = trt_common_->getBindingDimensions(0);
            bool const hasRuntimeDim = std::any_of(
                    input_dims.d, input_dims.d + input_dims.nbDims,
                    [](int32_t input_dim) { return input_dim == -1; });
            if (hasRuntimeDim) {
                input_dims.d[0] = batch_size_;
            }
            if (!image_buf_h_) {
                trt_common_->setBindingDimensions(0, input_dims);
                scales_.clear();
            }
            const float input_height = static_cast<float>(input_dims.d[2]);
            const float input_width = static_cast<float>(input_dims.d[3]);
            if (!image_buf_h_) {
                const float scale = std::min(input_width / width, input_height / height);
                for (int b = 0; b < batch_size_; b++) {
                    scales_.emplace_back(scale);
                }
                image_buf_h_ = cuda_utils::make_unique_host<unsigned char[]>(
                        width * height * 3 * batch_size_, cudaHostAllocWriteCombined);
                image_buf_d_ = cuda_utils::make_unique<unsigned char[]>(width * height * 3 * batch_size_);
            }
            if (multitask_) {
                size_t argmax_out_elem_num = 0;
                for (int m = 0; m < multitask_; m++) {
                    const auto output_dims =
                            trt_common_->getBindingDimensions(m + 2);  // 0 : input, 1 : output for detections
                    const float scale = std::min(
                            output_dims.d[3] / static_cast<float>(width),
                            output_dims.d[2] / static_cast<float>(height));
                    int out_w = static_cast<int>(width * scale);
                    int out_h = static_cast<int>(height * scale);
                    // size_t out_elem_num = std::accumulate(
                    // output_dims.d + 1, output_dims.d + output_dims.nbDims, 1, std::multiplies<int>());
                    // out_elem_num = out_elem_num * batch_size_;
                    size_t out_elem_num = out_w * out_h * batch_size_;
                    argmax_out_elem_num += out_elem_num;
                }
                argmax_buf_h_ =
                        cuda_utils::make_unique_host<unsigned char[]>(argmax_out_elem_num, cudaHostAllocPortable);
                argmax_buf_d_ = cuda_utils::make_unique<unsigned char[]>(argmax_out_elem_num);
            }
        }
    }

    void TrtRTMDet::printProfiling() {
        trt_common_->printProfiling();
    }

    void TrtRTMDet::preprocessGpu(const std::vector<cv::Mat> &images) {
        const auto batch_size = images.size();
        auto input_dims = trt_common_->getBindingDimensions(0);

        input_dims.d[0] = batch_size;
        for (const auto &image: images) {
            // if size of source input has been changed...
            int width = image.cols;
            int height = image.rows;
            if (src_width_ != -1 || src_height_ != -1) {
                if (width != src_width_ || height != src_height_) {
                    // Free cuda memory to reallocate
                    if (image_buf_h_) {
                        image_buf_h_.reset();
                    }
                    if (image_buf_d_) {
                        image_buf_d_.reset();
                    }
                    if (argmax_buf_h_) {
                        argmax_buf_h_.reset();
                    }
                    if (argmax_buf_d_) {
                        argmax_buf_d_.reset();
                    }
                }
            }
            src_width_ = width;
            src_height_ = height;
        }
        if (!image_buf_h_) {
            trt_common_->setBindingDimensions(0, input_dims);
            scales_.clear();
        }
        const float input_height = static_cast<float>(input_dims.d[2]);
        const float input_width = static_cast<float>(input_dims.d[3]);
        int b = 0;
        size_t argmax_out_elem_num = 0;
        for (const auto &image: images) {
            if (!image_buf_h_) {
                const float scale = std::min(input_width / image.cols, input_height / image.rows);
                scales_.emplace_back(scale);
                image_buf_h_ = cuda_utils::make_unique_host<unsigned char[]>(
                        image.cols * image.rows * 3 * batch_size, cudaHostAllocWriteCombined);
                image_buf_d_ =
                        cuda_utils::make_unique<unsigned char[]>(image.cols * image.rows * 3 * batch_size);
            }
            int index = b * image.cols * image.rows * 3;
            // Copy into pinned memory
            memcpy(
                    image_buf_h_.get() + index, &image.data[0],
                    image.cols * image.rows * 3 * sizeof(unsigned char));
            b++;

            if (multitask_) {
                for (int m = 0; m < multitask_; m++) {
                    const auto output_dims =
                            trt_common_->getBindingDimensions(m + 2);  // 0: input, 1: output for detections
                    const float scale = std::min
                            output_dims.d[3] / static_cast<float>(image.cols),
                            output_dims.d[2] / static_cast<float>(image.rows));
                    int out_w = static_cast<int>(image.cols * scale);
                    int out_h = static_cast<int>(image.rows * scale);
                    argmax_out_elem_num += out_w * out_h * batch_size;
                }
            }
        }

        if (multitask_) {
            if (!argmax_buf_h_) {
                argmax_buf_h_ =
                        cuda_utils::make_unique_host<unsigned char[]>(argmax_out_elem_num, cudaHostAllocPortable);
            }
            if (!argmax_buf_d_) {
                argmax_buf_d_ = cuda_utils::make_unique<unsigned char[]>(argmax_out_elem_num);
            }
        }

        // Copy into device memory
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                image_buf_d_.get(), image_buf_h_.get(),
                images[0].cols * images[0].rows * 3 * batch_size * sizeof(unsigned char),
                cudaMemcpyHostToDevice, *stream_));
        // Preprocess on GPU
        resize_bilinear_letterbox_nhwc_to_nchw32_batch_gpu(
                input_d_.get(), image_buf_d_.get(), input_width, input_height, 3, images[0].cols,
                images[0].rows, 3, batch_size, static_cast<float>(norm_factor_), *stream_);
    }

    void TrtRTMDet::preprocess(const std::vector<cv::Mat> &images) {
        const auto batch_size = images.size();
        auto input_dims = trt_common_->getBindingDimensions(0);
        input_dims.d[0] = batch_size;
        trt_common_->setBindingDimensions(0, input_dims);
        const float input_height = static_cast<float>(input_dims.d[2]);
        const float input_width = static_cast<float>(input_dims.d[3]);
        std::vector<cv::Mat> dst_images;
        scales_.clear();
        bool letterbox = true;
        if (letterbox) {
            for (const auto &image: images) {
                cv::Mat dst_image;
                const float scale = std::min(input_width / image.cols, input_height / image.rows);
                scales_.emplace_back(scale);
                const auto scale_size = cv::Size(image.cols * scale, image.rows * scale);
                cv::resize(image, dst_image, scale_size, 0, 0, cv::INTER_CUBIC);
                const auto bottom = input_height - dst_image.rows;
                const auto right = input_width - dst_image.cols;
                copyMakeBorder(
                        dst_image, dst_image, 0, bottom, 0, right, cv::BORDER_CONSTANT, {114, 114, 114});
                dst_images.emplace_back(dst_image);
            }
        } else {
            for (const auto &image: images) {
                cv::Mat dst_image;
                const float scale = -1.0;
                scales_.emplace_back(scale);
                const auto scale_size = cv::Size(input_width, input_height);
                cv::resize(image, dst_image, scale_size, 0, 0, cv::INTER_CUBIC);
                dst_images.emplace_back(dst_image);
            }
        }
        const auto chw_images = cv::dnn::blobFromImages(
                dst_images, norm_factor_, cv::Size(), cv::Scalar(), false, false, CV_32F);

        const auto data_length = chw_images.total();
        input_h_.reserve(data_length);
        const auto flat = chw_images.reshape(1, data_length);
        input_h_ = chw_images.isContinuous() ? flat : flat.clone();
        CHECK_CUDA_ERROR(cudaMemcpy(
                input_d_.get(), input_h_.data(), input_h_.size() * sizeof(float), cudaMemcpyHostToDevice));
        // No Need for Sync
    }

    bool TrtRTMDet::doInference(
            const std::vector<cv::Mat> &images, ObjectArrays &objects, std::vector<cv::Mat> &masks,
            [[maybe_unused]] std::vector<cv::Mat> &color_masks) {
        if (!trt_common_->isInitialized()) {
            return false;
        }

        if (use_gpu_preprocess_) {
            std::cout << "use_gpu_preprocess_" << std::endl;
            preprocessGpu(images);
        } else {
            std::cout << "use_preprocess_" << std::endl;
            preprocess(images);
        }

        if (needs_output_decode_) {
            return feedforwardAndDecode(images, objects, masks, color_masks);
        } else {
            return feedforward(images, objects);
        }
    }

    void TrtRTMDet::preprocessWithRoiGpu(
            const std::vector<cv::Mat> &images, const std::vector<cv::Rect> &rois) {
        const auto batch_size = images.size();
        auto input_dims = trt_common_->getBindingDimensions(0);

        input_dims.d[0] = batch_size;
        for (const auto &image: images) {
            // if size of source input has been changed...
            int width = image.cols;
            int height = image.rows;
            if (src_width_ != -1 || src_height_ != -1) {
                if (width != src_width_ || height != src_height_) {
                    // Free cuda memory to reallocate
                    if (image_buf_h_) {
                        image_buf_h_.reset();
                    }
                    if (image_buf_d_) {
                        image_buf_d_.reset();
                    }
                }
            }
            src_width_ = width;
            src_height_ = height;
        }
        if (!image_buf_h_) {
            trt_common_->setBindingDimensions(0, input_dims);
        }
        const float input_height = static_cast<float>(input_dims.d[2]);
        const float input_width = static_cast<float>(input_dims.d[3]);
        int b = 0;
        scales_.clear();

        if (!roi_h_) {
            roi_h_ = cuda_utils::make_unique_host<Roi[]>(batch_size, cudaHostAllocWriteCombined);
            roi_d_ = cuda_utils::make_unique<Roi[]>(batch_size);
        }

        for (const auto &image: images) {
            const float scale = std::min(
                    input_width / static_cast<float>(rois[b].width),
                    input_height / static_cast<float>(rois[b].height));
            scales_.emplace_back(scale);
            if (!image_buf_h_) {
                image_buf_h_ = cuda_utils::make_unique_host<unsigned char[]>(
                        image.cols * image.rows * 3 * batch_size, cudaHostAllocWriteCombined);
                image_buf_d_ =
                        cuda_utils::make_unique<unsigned char[]>(image.cols * image.rows * 3 * batch_size);
            }
            int index = b * image.cols * image.rows * 3;
            // Copy into pinned memory
            // memcpy(&(m_h_img[index]), &image.data[0], image.cols * image.rows * 3 * sizeof(unsigned
            // char));
            memcpy(
                    image_buf_h_.get() + index, &image.data[0],
                    image.cols * image.rows * 3 * sizeof(unsigned char));
            roi_h_[b].x = rois[b].x;
            roi_h_[b].y = rois[b].y;
            roi_h_[b].w = rois[b].width;
            roi_h_[b].h = rois[b].height;
            b++;
        }
        // Copy into device memory
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                image_buf_d_.get(), image_buf_h_.get(),
                images[0].cols * images[0].rows * 3 * batch_size * sizeof(unsigned char),
                cudaMemcpyHostToDevice, *stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                roi_d_.get(), roi_h_.get(), batch_size * sizeof(Roi), cudaMemcpyHostToDevice, *stream_));
        crop_resize_bilinear_letterbox_nhwc_to_nchw32_batch_gpu(
                input_d_.get(), image_buf_d_.get(), input_width, input_height, 3, roi_d_.get(), images[0].cols,
                images[0].rows, 3, batch_size, static_cast<float>(norm_factor_), *stream_);
    }

    void TrtRTMDet::preprocessWithRoi(
            const std::vector<cv::Mat> &images, const std::vector<cv::Rect> &rois) {
        const auto batch_size = images.size();
        auto input_dims = trt_common_->getBindingDimensions(0);
        input_dims.d[0] = batch_size;
        trt_common_->setBindingDimensions(0, input_dims);
        const float input_height = static_cast<float>(input_dims.d[2]);
        const float input_width = static_cast<float>(input_dims.d[3]);
        std::vector<cv::Mat> dst_images;
        scales_.clear();
        bool letterbox = true;
        int b = 0;
        if (letterbox) {
            for (const auto &image: images) {
                cv::Mat dst_image;
                cv::Mat cropped = image(rois[b]);
                const float scale = std::min(
                        input_width / static_cast<float>(rois[b].width),
                        input_height / static_cast<float>(rois[b].height));
                scales_.emplace_back(scale);
                const auto scale_size = cv::Size(rois[b].width * scale, rois[b].height * scale);
                cv::resize(cropped, dst_image, scale_size, 0, 0, cv::INTER_CUBIC);
                const auto bottom = input_height - dst_image.rows;
                const auto right = input_width - dst_image.cols;
                copyMakeBorder(
                        dst_image, dst_image, 0, bottom, 0, right, cv::BORDER_CONSTANT, {114, 114, 114});
                dst_images.emplace_back(dst_image);
                b++;
            }
        } else {
            for (const auto &image: images) {
                cv::Mat dst_image;
                const float scale = -1.0;
                scales_.emplace_back(scale);
                const auto scale_size = cv::Size(input_width, input_height);
                cv::resize(image, dst_image, scale_size, 0, 0, cv::INTER_CUBIC);
                dst_images.emplace_back(dst_image);
            }
        }
        const auto chw_images = cv::dnn::blobFromImages(
                dst_images, norm_factor_, cv::Size(), cv::Scalar(), false, false, CV_32F);

        const auto data_length = chw_images.total();
        input_h_.reserve(data_length);
        const auto flat = chw_images.reshape(1, data_length);
        input_h_ = chw_images.isContinuous() ? flat : flat.clone();
        CHECK_CUDA_ERROR(cudaMemcpy(
                input_d_.get(), input_h_.data(), input_h_.size() * sizeof(float), cudaMemcpyHostToDevice));
        // No Need for Sync
    }

    void TrtRTMDet::multiScalePreprocessGpu(const cv::Mat &image, const std::vector<cv::Rect> &rois) {
        const auto batch_size = rois.size();
        auto input_dims = trt_common_->getBindingDimensions(0);

        input_dims.d[0] = batch_size;

        // if size of source input has been changed...
        int width = image.cols;
        int height = image.rows;
        if (src_width_ != -1 || src_height_ != -1) {
            if (width != src_width_ || height != src_height_) {
                // Free cuda memory to reallocate
                if (image_buf_h_) {
                    image_buf_h_.reset();
                }
                if (image_buf_d_) {
                    image_buf_d_.reset();
                }
            }
        }
        src_width_ = width;
        src_height_ = height;

        if (!image_buf_h_) {
            trt_common_->setBindingDimensions(0, input_dims);
        }
        const float input_height = static_cast<float>(input_dims.d[2]);
        const float input_width = static_cast<float>(input_dims.d[3]);

        scales_.clear();

        if (!roi_h_) {
            roi_h_ = cuda_utils::make_unique_host<Roi[]>(batch_size, cudaHostAllocWriteCombined);
            roi_d_ = cuda_utils::make_unique<Roi[]>(batch_size);
        }

        for (size_t b = 0; b < rois.size(); b++) {
            const float scale = std::min(
                    input_width / static_cast<float>(rois[b].width),
                    input_height / static_cast<float>(rois[b].height));
            scales_.emplace_back(scale);
            roi_h_[b].x = rois[b].x;
            roi_h_[b].y = rois[b].y;
            roi_h_[b].w = rois[b].width;
            roi_h_[b].h = rois[b].height;
        }
        if (!image_buf_h_) {
            image_buf_h_ = cuda_utils::make_unique_host<unsigned char[]>(
                    image.cols * image.rows * 3 * 1, cudaHostAllocWriteCombined);
            image_buf_d_ = cuda_utils::make_unique<unsigned char[]>(image.cols * image.rows * 3 * 1);
        }
        int index = 0 * image.cols * image.rows * 3;
        // Copy into pinned memory
        memcpy(
                image_buf_h_.get() + index, &image.data[0],
                image.cols * image.rows * 3 * sizeof(unsigned char));

        // Copy into device memory
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                image_buf_d_.get(), image_buf_h_.get(), image.cols * image.rows * 3 * 1 * sizeof(unsigned char),
                cudaMemcpyHostToDevice, *stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                roi_d_.get(), roi_h_.get(), batch_size * sizeof(Roi), cudaMemcpyHostToDevice, *stream_));
        multi_scale_resize_bilinear_letterbox_nhwc_to_nchw32_batch_gpu(
                input_d_.get(), image_buf_d_.get(), input_width, input_height, 3, roi_d_.get(), image.cols,
                image.rows, 3, batch_size, static_cast<float>(norm_factor_), *stream_);
    }

    void TrtRTMDet::multiScalePreprocess(const cv::Mat &image, const std::vector<cv::Rect> &rois) {
        const auto batch_size = rois.size();
        auto input_dims = trt_common_->getBindingDimensions(0);
        input_dims.d[0] = batch_size;
        trt_common_->setBindingDimensions(0, input_dims);
        const float input_height = static_cast<float>(input_dims.d[2]);
        const float input_width = static_cast<float>(input_dims.d[3]);
        std::vector<cv::Mat> dst_images;
        scales_.clear();

        for (const auto &roi: rois) {
            cv::Mat dst_image;
            cv::Mat cropped = image(roi);
            const float scale = std::min(
                    input_width / static_cast<float>(roi.width), input_height / static_cast<float>(roi.height));
            scales_.emplace_back(scale);
            const auto scale_size = cv::Size(roi.width * scale, roi.height * scale);
            cv::resize(cropped, dst_image, scale_size, 0, 0, cv::INTER_CUBIC);
            const auto bottom = input_height - dst_image.rows;
            const auto right = input_width - dst_image.cols;
            copyMakeBorder(dst_image, dst_image, 0, bottom, 0, right, cv::BORDER_CONSTANT, {114, 114, 114});
            dst_images.emplace_back(dst_image);
        }
        const auto chw_images = cv::dnn::blobFromImages(
                dst_images, norm_factor_, cv::Size(), cv::Scalar(), false, false, CV_32F);

        const auto data_length = chw_images.total();
        input_h_.reserve(data_length);
        const auto flat = chw_images.reshape(1, data_length);
        input_h_ = chw_images.isContinuous() ? flat : flat.clone();
        CHECK_CUDA_ERROR(cudaMemcpy(
                input_d_.get(), input_h_.data(), input_h_.size() * sizeof(float), cudaMemcpyHostToDevice));
        // No Need for Sync
    }

    bool TrtRTMDet::doInferenceWithRoi(
            const std::vector<cv::Mat> &images, ObjectArrays &objects, const std::vector<cv::Rect> &rois) {
        std::vector<cv::Mat> masks;
        std::vector<cv::Mat> color_masks;
        if (!trt_common_->isInitialized()) {
            return false;
        }
        if (use_gpu_preprocess_) {
            preprocessWithRoiGpu(images, rois);
        } else {
            preprocessWithRoi(images, rois);
        }

        if (needs_output_decode_) {
            return feedforwardAndDecode(images, objects, masks, color_masks);
        } else {
            return feedforward(images, objects);
        }
    }

    bool TrtRTMDet::doMultiScaleInference(
            const cv::Mat &image, ObjectArrays &objects, const std::vector<cv::Rect> &rois) {
        std::vector<cv::Mat> images;
        if (!trt_common_->isInitialized()) {
            return false;
        }
        if (use_gpu_preprocess_) {
            multiScalePreprocessGpu(image, rois);
        } else {
            multiScalePreprocess(image, rois);
        }
        if (needs_output_decode_) {
            return multiScaleFeedforwardAndDecode(image, rois.size(), objects);
        } else {
            return multiScaleFeedforward(image, rois.size(), objects);
        }
    }

    bool TrtRTMDet::feedforward(const std::vector<cv::Mat> &images, ObjectArrays &objects) {
        std::vector<void *> buffers = {
                input_d_.get(), out_dets_d_.get(), out_labels_d_.get(), out_masks_d_.get()};

        trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

        const auto batch_size = images.size();
        auto out_dets = std::make_unique<float>(batch_size * 500);
        auto out_labels = std::make_unique<int32_t[]>(batch_size * 100);
        auto out_masks = std::make_unique<float[]>(batch_size * 640 * 640 * 100);

        std::cout << "111111111111111" << std::endl;

        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_dets.get(), out_dets_d_.get(), sizeof(float) * batch_size * 500,
                cudaMemcpyDeviceToHost, *stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_labels.get(), out_labels_d_.get(), sizeof(int32_t) * batch_size * 100,
                cudaMemcpyDeviceToHost, *stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_masks.get(), out_masks_d_.get(), sizeof(float) * batch_size * 640 * 640 * 100,
                cudaMemcpyDeviceToHost, *stream_));

        std::cout << "222222222222222222" << std::endl;

        for (int i = 0; i < 100; ++i) {
            std::cout << out_labels[i] << std::endl;
        }

        cudaStreamSynchronize(*stream_);

        std::cout << "3333333333" << std::endl;

        objects.clear();
//        for (size_t i = 0; i < batch_size; ++i) {
//            const size_t num_detection = static_cast<size_t>(out_num_detections[i]);
//            ObjectArray object_array;
//            object_array.reserve(num_detection);
//            for (size_t j = 0; j < num_detection; ++j) {
//                Object object{};
//                const auto x1 = out_boxes[i * max_detections_ * 4 + j * 4] / scales_[i];
//                const auto y1 = out_boxes[i * max_detections_ * 4 + j * 4 + 1] / scales_[i];
//                const auto x2 = out_boxes[i * max_detections_ * 4 + j * 4 + 2] / scales_[i];
//                const auto y2 = out_boxes[i * max_detections_ * 4 + j * 4 + 3] / scales_[i];
//                object.x_offset = std::clamp(0, static_cast<int32_t>(x1), images[i].cols);
//                object.y_offset = std::clamp(0, static_cast<int32_t>(y1), images[i].rows);
//                object.width = static_cast<int32_t>(std::max(0.0F, x2 - x1));
//                object.height = static_cast<int32_t>(std::max(0.0F, y2 - y1));
//                object.score = out_scores[i * max_detections_ + j];
//                object.type = out_classes[i * max_detections_ + j];
//                object_array.emplace_back(object);
//            }
//            objects.emplace_back(object_array);
//        }
        return true;
    }

    bool TrtRTMDet::feedforwardAndDecode(
            const std::vector<cv::Mat> &images, ObjectArrays &objects, std::vector<cv::Mat> &out_masks,
            [[maybe_unused]] std::vector<cv::Mat> &color_masks) {
        std::vector<void *> buffers = {input_d_.get(), out_prob_d_.get()};
        if (multitask_) {
            buffers = {input_d_.get(), out_prob_d_.get(), segmentation_out_prob_d_.get()};
        }
        trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

        const auto batch_size = images.size();

        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_prob_h_.get(), out_prob_d_.get(), sizeof(float) * out_elem_num_, cudaMemcpyDeviceToHost,
                *stream_));
        if (multitask_) {
            CHECK_CUDA_ERROR(cudaMemcpyAsync(
                    segmentation_out_prob_h_.get(), segmentation_out_prob_d_.get(),
                    sizeof(float) * segmentation_out_elem_num_, cudaMemcpyDeviceToHost, *stream_));
        }
        cudaStreamSynchronize(*stream_);
        objects.clear();

        for (size_t i = 0; i < batch_size; ++i) {
            auto image_size = images[i].size();
            float *batch_prob = out_prob_h_.get() + (i * out_elem_num_per_batch_);
            ObjectArray object_array;
            decodeOutputs(batch_prob, object_array, scales_[i], image_size);
            // add refine mask using object
            objects.emplace_back(object_array);
            if (multitask_) {
                segmentation_masks_.clear();
                float *segmentation_results =
                        segmentation_out_prob_h_.get() + (i * segmentation_out_elem_num_per_batch_);
                size_t counter = 0;
                int batch =
                        static_cast<int>(segmentation_out_elem_num_ / segmentation_out_elem_num_per_batch_);
                for (int m = 0; m < multitask_; m++) {
                    const auto output_dims =
                            trt_common_->getBindingDimensions(m + 2);  // 0 : input, 1 : output for detections
                    size_t out_elem_num = std::accumulate(
                            output_dims.d + 1, output_dims.d + output_dims.nbDims, 1, std::multiplies<int>());
                    out_elem_num = out_elem_num * batch;
                    const float scale = std::min(
                            output_dims.d[3] / static_cast<float>(image_size.width),
                            output_dims.d[2] / static_cast<float>(image_size.height));
                    int out_w = static_cast<int>(image_size.width * scale);
                    int out_h = static_cast<int>(image_size.height * scale);
                    cv::Mat mask;
                    if (use_gpu_preprocess_) {
                        float *d_segmentation_results =
                                segmentation_out_prob_d_.get() + (i * segmentation_out_elem_num_per_batch_);
                        mask = getMaskImageGpu(&(d_segmentation_results[counter]), output_dims, out_w, out_h, i);
                    } else {
                        mask = getMaskImage(&(segmentation_results[counter]), output_dims, out_w, out_h);
                    }
                    segmentation_masks_.emplace_back(std::move(mask));
                    counter += out_elem_num;
                }
                // semantic segmentation was fixed as first task
                out_masks.at(i) = segmentation_masks_.at(0);
            } else {
                continue;
            }
        }
        return true;
    }

// This method is assumed to be called when specified YOLOX model contains
// EfficientNMS_TRT module.
    bool TrtRTMDet::multiScaleFeedforward(const cv::Mat &image, int batch_size, ObjectArrays &objects) {
        std::vector<void *> buffers = {
                input_d_.get(), out_dets_d_.get(), out_labels_d_.get(), out_masks_d_.get()};

        trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

        auto out_num_detections = std::make_unique<int32_t[]>(batch_size);
        auto out_boxes = std::make_unique<float[]>(4 * batch_size * max_detections_);
        auto out_scores = std::make_unique<float[]>(batch_size * max_detections_);
        auto out_classes = std::make_unique<int32_t[]>(batch_size * max_detections_);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_num_detections.get(), out_dets_d_.get(), sizeof(float) * batch_size * 500,
                cudaMemcpyDeviceToHost, *stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_boxes.get(), out_labels_d_.get(), sizeof(int32_t) * batch_size * 100,
                cudaMemcpyDeviceToHost, *stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_scores.get(), out_masks_d_.get(), sizeof(float) * batch_size * 640 * 640 * 100,
                cudaMemcpyDeviceToHost, *stream_));
        cudaStreamSynchronize(*stream_);

        objects.clear();
        for (int i = 0; i < batch_size; ++i) {
            const size_t num_detection = static_cast<size_t>(out_num_detections[i]);
            ObjectArray object_array(num_detection);
            for (size_t j = 0; j < num_detection; ++j) {
                Object object{};
                const auto x1 = out_boxes[i * max_detections_ * 4 + j * 4] / scales_[i];
                const auto y1 = out_boxes[i * max_detections_ * 4 + j * 4 + 1] / scales_[i];
                const auto x2 = out_boxes[i * max_detections_ * 4 + j * 4 + 2] / scales_[i];
                const auto y2 = out_boxes[i * max_detections_ * 4 + j * 4 + 3] / scales_[i];
                object.x_offset = std::clamp(0, static_cast<int32_t>(x1), image.cols);
                object.y_offset = std::clamp(0, static_cast<int32_t>(y1), image.rows);
                object.width = static_cast<int32_t>(std::max(0.0F, x2 - x1));
                object.height = static_cast<int32_t>(std::max(0.0F, y2 - y1));
                object.score = out_scores[i * max_detections_ + j];
                object.type = out_classes[i * max_detections_ + j];
                object_array.emplace_back(object);
            }
            objects.emplace_back(object_array);
        }
        return true;
    }

    bool TrtRTMDet::multiScaleFeedforwardAndDecode(
            const cv::Mat &image, int batch_size, ObjectArrays &objects) {
        std::vector<void *> buffers = {input_d_.get(), out_prob_d_.get()};
        trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_prob_h_.get(), out_prob_d_.get(), sizeof(float) * out_elem_num_, cudaMemcpyDeviceToHost,
                *stream_));
        cudaStreamSynchronize(*stream_);
        objects.clear();

        for (int i = 0; i < batch_size; ++i) {
            auto image_size = image.size();
            float *batch_prob = out_prob_h_.get() + (i * out_elem_num_per_batch_);
            ObjectArray object_array;
            decodeOutputs(batch_prob, object_array, scales_[i], image_size);
            objects.emplace_back(object_array);
        }
        return true;
    }

    void TrtRTMDet::decodeOutputs(
            float *prob, ObjectArray &objects, float scale, cv::Size &img_size) const {
        ObjectArray proposals;
        auto input_dims = trt_common_->getBindingDimensions(0);
        const float input_height = static_cast<float>(input_dims.d[2]);
        const float input_width = static_cast<float>(input_dims.d[3]);
        std::vector<GridAndStride> grid_strides;
        generateGridsAndStride(input_width, input_height, output_strides_, grid_strides);
        generateYoloxProposals(grid_strides, prob, score_threshold_, proposals);

        qsortDescentInplace(proposals);

        std::vector<int> picked;
        // cspell: ignore Bboxes
        nmsSortedBboxes(proposals, picked, nms_threshold_);

        int count = static_cast<int>(picked.size());
        objects.resize(count);
        float scale_x = input_width / static_cast<float>(img_size.width);
        float scale_y = input_height / static_cast<float>(img_size.height);
        for (int i = 0; i < count; i++) {
            objects[i] = proposals[picked[i]];

            float x0, y0, x1, y1;
            // adjust offset to original unpadded
            if (scale == -1.0) {
                x0 = (objects[i].x_offset) / scale_x;
                y0 = (objects[i].y_offset) / scale_y;
                x1 = (objects[i].x_offset + objects[i].width) / scale_x;
                y1 = (objects[i].y_offset + objects[i].height) / scale_y;
            } else {
                x0 = (objects[i].x_offset) / scale;
                y0 = (objects[i].y_offset) / scale;
                x1 = (objects[i].x_offset + objects[i].width) / scale;
                y1 = (objects[i].y_offset + objects[i].height) / scale;
            }
            // clip
            x0 = std::clamp(x0, 0.f, static_cast<float>(img_size.width - 1));
            y0 = std::clamp(y0, 0.f, static_cast<float>(img_size.height - 1));
            x1 = std::clamp(x1, 0.f, static_cast<float>(img_size.width - 1));
            y1 = std::clamp(y1, 0.f, static_cast<float>(img_size.height - 1));

            objects[i].x_offset = x0;
            objects[i].y_offset = y0;
            objects[i].width = x1 - x0;
            objects[i].height = y1 - y0;
        }
    }

    void TrtRTMDet::generateGridsAndStride(
            const int target_w, const int target_h, const std::vector<int> &strides,
            std::vector<GridAndStride> &grid_strides) const {
        for (auto stride: strides) {
            int num_grid_w = target_w / stride;
            int num_grid_h = target_h / stride;
            for (int g1 = 0; g1 < num_grid_h; g1++) {
                for (int g0 = 0; g0 < num_grid_w; g0++) {
                    grid_strides.push_back(GridAndStride{g0, g1, stride});
                }
            }
        }
    }

    void TrtRTMDet::generateYoloxProposals(
            std::vector<GridAndStride> grid_strides, float *feat_blob, float prob_threshold,
            ObjectArray &objects) const {
        const int num_anchors = grid_strides.size();

        for (int anchor_idx = 0; anchor_idx < num_anchors; anchor_idx++) {
            const int grid0 = grid_strides[anchor_idx].grid0;
            const int grid1 = grid_strides[anchor_idx].grid1;
            const int stride = grid_strides[anchor_idx].stride;

            const int basic_pos = anchor_idx * (num_class_ + 5);

            // yolox/models/yolo_head.py decode logic
            // To apply this logic, YOLOX head must output raw value
            // (i.e., `decode_in_inference` should be False)
            float x_center = (feat_blob[basic_pos + 0] + grid0) * stride;
            float y_center = (feat_blob[basic_pos + 1] + grid1) * stride;

            // exp is complex for embedded processors
            // float w = exp(feat_blob[basic_pos + 2]) * stride;
            // float h = exp(feat_blob[basic_pos + 3]) * stride;
            // float x0 = x_center - w * 0.5f;
            // float y0 = y_center - h * 0.5f;

            float box_objectness = feat_blob[basic_pos + 4];
            for (int class_idx = 0; class_idx < num_class_; class_idx++) {
                float box_cls_score = feat_blob[basic_pos + 5 + class_idx];
                float box_prob = box_objectness * box_cls_score;
                if (box_prob > prob_threshold) {
                    Object obj;
                    // On-demand applying for exp
                    float w = exp(feat_blob[basic_pos + 2]) * stride;
                    float h = exp(feat_blob[basic_pos + 3]) * stride;
                    float x0 = x_center - w * 0.5f;
                    float y0 = y_center - h * 0.5f;
                    obj.x_offset = x0;
                    obj.y_offset = y0;
                    obj.height = h;
                    obj.width = w;
                    obj.type = class_idx;
                    obj.score = box_prob;

                    objects.push_back(obj);
                }
            }  // class loop
        }    // point anchor loop
    }

    void TrtRTMDet::qsortDescentInplace(ObjectArray &face_objects, int left, int right) const {
        int i = left;
        int j = right;
        float p = face_objects[(left + right) / 2].score;

        while (i <= j) {
            while (face_objects[i].score > p) {
                i++;
            }

            while (face_objects[j].score < p) {
                j--;
            }

            if (i <= j) {
                // swap
                std::swap(face_objects[i], face_objects[j]);

                i++;
                j--;
            }
        }

#pragma omp parallel sections
        {
#pragma omp section
            {
                if (left < j) {
                    qsortDescentInplace(face_objects, left, j);
                }
            }
#pragma omp section
            {
                if (i < right) {
                    qsortDescentInplace(face_objects, i, right);
                }
            }
        }
    }

    void TrtRTMDet::nmsSortedBboxes(
            const ObjectArray &face_objects, std::vector<int> &picked, float nms_threshold) const {
        picked.clear();
        const int n = face_objects.size();

        std::vector<float> areas(n);
        for (int i = 0; i < n; i++) {
            cv::Rect rect(
                    face_objects[i].x_offset, face_objects[i].y_offset, face_objects[i].width,
                    face_objects[i].height);
            areas[i] = rect.area();
        }

        for (int i = 0; i < n; i++) {
            const Object &a = face_objects[i];

            int keep = 1;
            for (int j = 0; j < static_cast<int>(picked.size()); j++) {
                const Object &b = face_objects[picked[j]];

                // intersection over union
                float inter_area = intersectionArea(a, b);
                float union_area = areas[i] + areas[picked[j]] - inter_area;
                // float IoU = inter_area / union_area
                if (inter_area / union_area > nms_threshold) {
                    keep = 0;
                }
            }

            if (keep) {
                picked.push_back(i);
            }
        }
    }

    cv::Mat TrtRTMDet::getMaskImageGpu(float *d_prob, nvinfer1::Dims dims, int out_w, int out_h, int b) {
        // NCHW
        int classes = dims.d[1];
        int height = dims.d[2];
        int width = dims.d[3];
        cv::Mat mask = cv::Mat::zeros(out_h, out_w, CV_8UC1);
        int index = b * out_w * out_h;
        argmax_gpu(
                (unsigned char *) argmax_buf_d_.get() + index, d_prob, out_w, out_h, width, height, classes, 1,
                *stream_);
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                argmax_buf_h_.get(), argmax_buf_d_.get(), sizeof(unsigned char) * 1 * out_w * out_h,
                cudaMemcpyDeviceToHost, *stream_));
        cudaStreamSynchronize(*stream_);
        std::memcpy(mask.data, argmax_buf_h_.get() + index, sizeof(unsigned char) * 1 * out_w * out_h);
        return mask;
    }

    cv::Mat TrtRTMDet::getMaskImage(float *prob, nvinfer1::Dims dims, int out_w, int out_h) {
        // NCHW
        int classes = dims.d[1];
        int height = dims.d[2];
        int width = dims.d[3];
        cv::Mat mask = cv::Mat::zeros(out_h, out_w, CV_8UC1);
        // argmax
        // #pragma omp parallel for
        for (int y = 0; y < out_h; y++) {
            for (int x = 0; x < out_w; x++) {
                float max = 0.0;
                int index = 0;
                for (int c = 0; c < classes; c++) {
                    float value = prob[c * height * width + y * width + x];
                    if (max < value) {
                        max = value;
                        index = c;
                    }
                }
                mask.at<unsigned char>(y, x) = index;
            }
        }
        return mask;
    }

    int TrtRTMDet::getMultitaskNum(void) {
        return multitask_;
    }

    void TrtRTMDet::getColorizedMask(
            const std::vector<tensorrt_rtmdet::Colormap> &colormap, const cv::Mat &mask, cv::Mat &cmask) {
        int width = mask.cols;
        int height = mask.rows;
        if ((cmask.cols != mask.cols) || (cmask.rows != mask.rows)) {
            throw std::runtime_error("input and output image have difference size.");
            return;
        }
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                unsigned char id = mask.at<unsigned char>(y, x);
                cmask.at<cv::Vec3b>(y, x)[0] = colormap[id].color[2];
                cmask.at<cv::Vec3b>(y, x)[1] = colormap[id].color[1];
                cmask.at<cv::Vec3b>(y, x)[2] = colormap[id].color[0];
            }
        }
    }

}