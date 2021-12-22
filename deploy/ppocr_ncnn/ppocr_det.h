//
// Created by tjm on 2021/12/18.
//

#ifndef PPOCR_PPOCR_DET_H
#define PPOCR_PPOCR_DET_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"
//#include "layer.h"


typedef struct BoxInfo{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
} BoxInfo;


class OCRTextDet{

public:
    OCRTextDet(const char* param, const char* bin);
    ~OCRTextDet();
    ncnn::Net* net{};
    cv::Mat detector(const cv::Mat& image);
//    friend void display(CenterDet *cdt);

private:
    cv::Mat preprocess(const cv::Mat& image, ncnn::Mat& in, bool org_size=false, bool keep_ratio=false);
//    void decode(const ncnn::Mat& heatmap, const ncnn::Mat& reg_box, std::vector<BoxInfo>& results, float score, float nms) const;
    void resize(const cv::Mat &img, cv::Mat &resize_img, float &ratio_w, float &ratio_h) const;
//    static void draw_bboxes(const cv::Mat& image, const std::vector<BoxInfo>& bboxes);
    static cv::Mat resize(const cv::Mat& image, const cv::Size_<int>& outsize);
//    static void nms(std::vector<BoxInfo>& bboxes, float nms_thr);
    void postprocess(const cv::Mat& src);

    int max_side_len = 960;
    float thresh = 0.3;
    float box_thresh = 0.5;
    float unclip_ratio = 1.5;
    cv::Size_<int> in_size = cv::Size(480, 480);
//    const float mean_vals[3] = {0.485f, 0.456f, 0.406f};
//    const float norm_vals[3] = {1/0.229f, 1/0.224f, 1/0.225f};
    const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    const float norm_vals[3] = {0.017f, 0.017f, 0.017f};

};


template <class T> inline T clamp(T x, T min, T max) {
    if (x > max)
        return max;
    if (x < min)
        return min;
    return x;
}



#endif //PPOCR_PPOCR_DET_H
