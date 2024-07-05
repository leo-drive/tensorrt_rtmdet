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
                         const size_t max_workspace_size, [[maybe_unused]] const std::string &color_map_path,
                         const std::vector<std::string> &plugin_paths) {
        RCLCPP_INFO(rclcpp::get_logger("tensorrt_rtmdet"), "tensorrt_rtmdet has been started.");

        src_width_ = -1;
        src_height_ = -1;
        norm_factor_ = norm_factor;
        batch_size_ = batch_config[2];
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
                    model_path, precision, std::move(calibrator), batch_config, max_workspace_size, build_config,
                    plugin_paths);
        } else {
            trt_common_ = std::make_unique<tensorrt_common::TrtCommon>(
                    model_path, precision, nullptr, batch_config, max_workspace_size, build_config, plugin_paths);
        }
        trt_common_->setup();

        if (!trt_common_->isInitialized()) {
            return;
        }

        num_class_ = num_class;
        score_threshold_ = score_threshold;
        nms_threshold_ = nms_threshold;

        const auto input_dims = trt_common_->getBindingDimensions(0);
        const auto out_scores_dims = trt_common_->getBindingDimensions(3);

        max_detections_ = out_scores_dims.d[1];
        model_input_height_ = input_dims.d[2];
        model_input_width_ = input_dims.d[3];

        input_d_ = cuda_utils::make_unique<float[]>(batch_size_ * input_dims.d[1] * input_dims.d[2] * input_dims.d[3]);
        out_dets_d_ = cuda_utils::make_unique<float[]>(batch_size_ * max_detections_ * 5);
        out_labels_d_ = cuda_utils::make_unique<int32_t[]>(batch_size_ * max_detections_);
        out_masks_d_ = cuda_utils::make_unique<float[]>(
                batch_size_ * max_detections_ * model_input_width_ * model_input_height_);

        if (use_gpu_preprocess) {
            use_gpu_preprocess_ = true;
            image_buf_h_ = nullptr;
            image_buf_d_ = nullptr;
        } else {
            use_gpu_preprocess_ = false;
        }

        // Segmentation
        readColorMap(color_map_path);
    }

    TrtRTMDet::~TrtRTMDet() {
        if (use_gpu_preprocess_) {
            if (image_buf_h_) {
                image_buf_h_.reset();
            }
            if (image_buf_d_) {
                image_buf_d_.reset();
            }
        }
    }

    void TrtRTMDet::printProfiling(void) {
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
                }
            }
            src_width_ = width;
            src_height_ = height;
        }
        if (!image_buf_h_) {
            trt_common_->setBindingDimensions(0, input_dims);
            scale_width_ = 0;
            scale_height_ = 0;
        }
        const float input_height = static_cast<float>(input_dims.d[2]);
        const float input_width = static_cast<float>(input_dims.d[3]);
        int b = 0;
        for (const auto &image: images) {
            if (!image_buf_h_) {
                scale_width_ = input_width / image.cols;
                scale_height_ = input_height / image.rows;
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
        std::vector<cv::Mat> dst_images;
        bool letterbox = true;
        if (letterbox) {
            for (const auto &image: images) {
                cv::Mat dst_image;
                cv::resize(image, dst_image, cv::Size(640, 640));
                dst_image.convertTo(dst_image, CV_32F);
                dst_image -= cv::Scalar(103.53, 116.28, 123.675);
                dst_image /= cv::Scalar(57.375, 57.12, 58.395);
                dst_images.emplace_back(dst_image);
            }
        } else {
            for (const auto &image: images) {
                cv::Mat dst_image;
                const auto scale_size = cv::Size(model_input_width_, model_input_height_);
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

    bool TrtRTMDet::doInference(const std::vector<cv::Mat> &images, ObjectArrays &objects) {
        if (!trt_common_->isInitialized()) {
            return false;
        }

        if (use_gpu_preprocess_) {
            preprocessGpu(images);
        } else {
            preprocess(images);
        }
        return feedforward(images, objects);
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

        if (!roi_h_) {
            roi_h_ = cuda_utils::make_unique_host<Roi[]>(batch_size, cudaHostAllocWriteCombined);
            roi_d_ = cuda_utils::make_unique<Roi[]>(batch_size);
        }

        for (const auto &image: images) {
            scale_width_ = input_width / static_cast<float>(rois[b].width);
            scale_height_ = input_height / static_cast<float>(rois[b].height);
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
        bool letterbox = true;
        int b = 0;
        if (letterbox) {
            for (const auto &image: images) {
                cv::Mat dst_image;
                cv::Mat cropped = image(rois[b]);
                scale_width_ = input_width / static_cast<float>(rois[b].width);
                scale_height_ = input_height / static_cast<float>(rois[b].height);
                const auto scale_size = cv::Size(rois[b].width * scale_width_, rois[b].height * scale_height_);
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
                const auto scale_size = cv::Size(model_input_width_, model_input_height_);
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

        if (!roi_h_) {
            roi_h_ = cuda_utils::make_unique_host<Roi[]>(batch_size, cudaHostAllocWriteCombined);
            roi_d_ = cuda_utils::make_unique<Roi[]>(batch_size);
        }

        for (size_t b = 0; b < rois.size(); b++) {
            scale_width_ = input_width / static_cast<float>(rois[b].width);
            scale_height_ = input_height / static_cast<float>(rois[b].height);
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

        for (const auto &roi: rois) {
            cv::Mat dst_image;
            cv::Mat cropped = image(roi);
            scale_width_ = input_width / static_cast<float>(roi.width);
            scale_height_ = input_height / static_cast<float>(roi.height);
            const auto scale_size = cv::Size(roi.width * scale_width_, roi.height * scale_height_);
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
        if (!trt_common_->isInitialized()) {
            return false;
        }
        if (use_gpu_preprocess_) {
            preprocessWithRoiGpu(images, rois);
        } else {
            preprocessWithRoi(images, rois);
        }
        return feedforward(images, objects);
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
        return multiScaleFeedforward(image, rois.size(), objects);
    }

    bool TrtRTMDet::feedforward([[maybe_unused]] const std::vector<cv::Mat> &images,
                                [[maybe_unused]] ObjectArrays &objects) {
        std::vector<void *> buffers = {
                input_d_.get(), out_dets_d_.get(), out_labels_d_.get(), out_masks_d_.get()};

        trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

        const auto batch_size = images.size();
        auto out_dets = std::make_unique<float[]>(batch_size_ * max_detections_ * 5);
        auto out_labels = std::make_unique<int32_t[]>(1 * max_detections_);
        auto out_masks = std::make_unique<float[]>(1 * max_detections_ * model_input_width_ * model_input_height_);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_dets.get(), out_dets_d_.get(), sizeof(float) * batch_size_ * max_detections_ * 5,
                cudaMemcpyDeviceToHost, *stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_labels.get(), out_labels_d_.get(), sizeof(int32_t) * batch_size_ * max_detections_,
                cudaMemcpyDeviceToHost, *stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_masks.get(), out_masks_d_.get(),
                sizeof(float) * batch_size_ * max_detections_ * model_input_width_ * model_input_height_,
                cudaMemcpyDeviceToHost, *stream_));

        cudaStreamSynchronize(*stream_);

        // POST PROCESSING

        // VISUALIZATION
        for (size_t i = 0; i < batch_size; ++i) {
            cv::Mat output_image = images[i].clone();
            for (int index = 0; index < 100; ++index) {
                if (out_dets[(5 * index) + 4] < score_threshold_) {
                    break;
                }
                std::cout << "score: " << out_dets[(5 * index) + 4] << " " << "label: " << out_labels[index]
                          << std::endl;

                cv::Mat mask(model_input_width_, model_input_height_, CV_32F,
                             out_masks.get() + index * model_input_width_ * model_input_height_);

                double minVal, maxVal;
                cv::minMaxLoc(mask, &minVal, &maxVal);
                mask.convertTo(mask, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));

                cv::resize(mask, mask, cv::Size(output_image.cols, output_image.rows));

                auto processPixel = [&](cv::Vec3b &pixel, const int *position) -> void {
                    int i = position[0];
                    int j = position[1];

                    if (mask.at<uchar>(i, j) > 200) {
                        cv::Vec3b color(
                                color_map_[out_labels[index]].color[0] * 0.5 + pixel[0] * 0.5,
                                color_map_[out_labels[index]].color[1] * 0.5 + pixel[1] * 0.5,
                                color_map_[out_labels[index]].color[2] * 0.5 + pixel[2] * 0.5
                        );
                        // TODO: check if color greater than 255
                        pixel = color;
                    }
                };
                output_image.forEach<cv::Vec3b>(processPixel);

                // Draw rectangle around the object
                cv::rectangle(output_image,
                              cv::Point(static_cast<int>(out_dets[(5 * index) + 0] * (1 / scale_width_)),
                                        static_cast<int>(out_dets[(5 * index) + 1] * (1 / scale_height_))),
                              cv::Point(static_cast<int>(out_dets[(5 * index) + 2] * (1 / scale_width_)),
                                        static_cast<int>(out_dets[(5 * index) + 3] * (1 / scale_height_))),
                              color_map_[out_labels[index]].color, 2);
                // Write the class name
                cv::putText(output_image, color_map_[out_labels[index]].label,
                            cv::Point(static_cast<int>(out_dets[(5 * index) + 0] * (1 / scale_width_)),
                                      static_cast<int>(out_dets[(5 * index) + 1] * (1 / scale_height_))),
                            cv::FONT_HERSHEY_SIMPLEX, 1, color_map_[out_labels[index]].color, 2);
            }

            cv::Mat resized_image;
            cv::resize(output_image, resized_image, cv::Size(1280, 720));
            cv::imshow("output", resized_image);
            cv::waitKey(1);

        }

        return true;
    }

    bool TrtRTMDet::multiScaleFeedforward([[maybe_unused]] const cv::Mat &image, int batch_size,
                                          [[maybe_unused]] ObjectArrays &objects) {
        std::vector<void *> buffers = {
                input_d_.get(), out_dets_d_.get(), out_labels_d_.get(), out_masks_d_.get()};

        trt_common_->enqueueV2(buffers.data(), *stream_, nullptr);

        auto out_dets = std::make_unique<float[]>(1 * 100 * 5);
        auto out_labels = std::make_unique<int32_t[]>(1 * 100);
        auto out_masks = std::make_unique<float[]>(1 * 100 * 640 * 640);

        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_dets.get(), out_dets_d_.get(), sizeof(float) * 4 * batch_size * max_detections_,
                cudaMemcpyDeviceToHost, *stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_labels.get(), out_labels_d_.get(), sizeof(int32_t) * batch_size,
                cudaMemcpyDeviceToHost, *stream_));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(
                out_masks.get(), out_masks_d_.get(), sizeof(float) * batch_size * max_detections_,
                cudaMemcpyDeviceToHost, *stream_));
        cudaStreamSynchronize(*stream_);

        return true;
    }

    void TrtRTMDet::readColorMap(const std::string &color_map_path) {
        std::vector<std::string> color_list = loadListFromTextFile(color_map_path);
        for (int i = 1; i < static_cast<int>(color_list.size()); i++) {
            auto splitString = [](const std::string &str, char delimiter) -> std::vector<std::string> {
                std::vector<std::string> result;
                std::stringstream ss(str);
                std::string item;

                while (std::getline(ss, item, delimiter)) {
                    result.push_back(item);
                }

                return result;
            };
            std::vector<std::string> tokens = splitString(color_list[i], ',');

            LabelColor label_color;
            label_color.label = tokens[1];
            label_color.color[0] = std::stoi(tokens[2]);
            label_color.color[1] = std::stoi(tokens[3]);
            label_color.color[2] = std::stoi(tokens[4]);
            color_map_[std::stoi(tokens[0])] = label_color;
        }
    }
}