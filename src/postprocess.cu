#include "tensorrt_rtmdet/postprocess.hpp"

namespace tensorrt_rtmdet {
    __global__ void create_detections(float *out_dets_d, int *out_labels_d, float *output_mask_d,
                                      int detection_number, float score_threshold, Detection *detections) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (out_dets_d[idx * 5 + 4] < score_threshold) return;

        if (idx < detection_number) {
            Detection det;
            det.x1 = out_dets_d[idx * 5];
            det.y1 = out_dets_d[idx * 5 + 1];
            det.x2 = out_dets_d[idx * 5 + 2];
            det.y2 = out_dets_d[idx * 5 + 3];
            det.score = out_dets_d[idx * 5 + 4];
            det.label = out_labels_d[idx];
            det.mask = output_mask_d + idx * 640 * 640;
            detections[idx] = det;
//            printf("Detection %d: x1=%f, y1=%f, x2=%f, y2=%f, score=%f, label=%d\n", idx, det.x1, det.y1, det.x2,
//                   det.y2,
//                   det.score, det.label);
        }
    }

    __global__ void count_detections_by_score(int detection_counter_by_score, float *out_dets_d) {
        int idx = threadIdx.x + blockIdx.x * blockDim.x;

        if (out_dets_d[idx * 5 + 4] > 0.5) {
            detection_counter_by_score++;
        }
    }

    void create_detections_gpu(float *out_dets_d, int *out_labels_d, float *output_mask_d,
                               int detection_number, float score_threshold, Detection *detections) {
        int block_size = 512;
        int grid_size = 100;

        int detection_counter_by_score = 0;
        count_detections_by_score<<<grid_size, block_size>>>(detection_counter_by_score, out_dets_d);

        create_detections<<<grid_size, block_size>>>(out_dets_d, out_labels_d, output_mask_d, detection_number,
                                                     score_threshold, detections);

//        for (int i = 0; i < detection_number; i++) {
//            printf("Detection %d: x1=%f, y1=%f, x2=%f, y2=%f, score=%f, label=%d\n", i, detections[i].x1,
//                   detections[i].y1, detections[i].x2, detections[i].y2, detections[i].score, detections[i].label);
//        }
    }
}