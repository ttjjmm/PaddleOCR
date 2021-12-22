//
// Created by ubuntu on 2021/12/20.
//

#ifndef PPOCR_PPOCR_CLS_H
#define PPOCR_PPOCR_CLS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"


typedef struct ClsInfo{
    int label;
    float score;
} ClsInfo;

class OCRTextCls{

public:
    OCRTextCls(const char* param, const char* bin);
    ~OCRTextCls();
    ncnn::Net* net{};
    ClsInfo detector(const cv::Mat& image);
//    friend void display(CenterDet *cdt);

private:
    cv::Mat preprocess(const cv::Mat& image, ncnn::Mat& in, bool org_size=false, bool keep_ratio=false);
//    void decode(const ncnn::Mat& heatmap, const ncnn::Mat& reg_box, std::vector<BoxInfo>& results, float score, float nms) const;
//
//    static void draw_bboxes(const cv::Mat& image, const std::vector<BoxInfo>& bboxes);
    static cv::Mat resize(const cv::Mat& image, const cv::Size_<int>& outsize);
//    static void nms(std::vector<BoxInfo>& bboxes, float nms_thr);
    void postprocess(const cv::Mat& src);

    cv::Size_<int> in_size = cv::Size(192, 48);
//    const float mean_vals[3] = {0.f, 0.f, 0.f};
//    const float norm_vals[3] = {1/255.f, 1/255.f, 1/225.f};
    const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    const float norm_vals[3] = {0.017f, 0.017f, 0.017f};
    float thresh = 0.3;
};






#endif //PPOCR_PPOCR_CLS_H
