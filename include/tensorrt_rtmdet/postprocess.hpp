#include <iostream>
#include <vector>

namespace tensorrt_rtmdet {
    struct Detection {
        float x1, y1, x2, y2, score;
        int label;
        float *mask;
    };

//    using DetectionArray = std::vector<Detection>;

    extern void create_detections_gpu(float *out_dets_d, int *out_labels_d, float *output_mask_d,
                                      int detection_number, float score_threshold, Detection *detections);
} // namespace tensorrt_rtmdet