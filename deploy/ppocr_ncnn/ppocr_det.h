//
// Created by tjm on 2021/12/18.
//

#ifndef PPOCR_PPOCR_DET_H
#define PPOCR_PPOCR_DET_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"
#include "layer.h"


typedef struct BoxInfo{
    float x1;
    float y1;
    float x2;
    float y2;
    float score;
    int label;
} BoxInfo;


class OCRTextDet{

public:
    OCRTextDet(const char* param, const char* bin);
    ~OCRTextDet();
    ncnn::Net* net{};
    std::vector<BoxInfo> detector(const cv::Mat& image);
//    friend void display(CenterDet *cdt);

private:
    cv::Mat preprocess(const cv::Mat& image, ncnn::Mat& in, bool org_size=false, bool keep_ratio=false);
//    void decode(const ncnn::Mat& heatmap, const ncnn::Mat& reg_box, std::vector<BoxInfo>& results, float score, float nms) const;
//
//    static void draw_bboxes(const cv::Mat& image, const std::vector<BoxInfo>& bboxes);
    static cv::Mat resize(const cv::Mat& image, const cv::Size_<int>& outsize);
//    static void nms(std::vector<BoxInfo>& bboxes, float nms_thr);


    cv::Size_<int> in_size = cv::Size(416, 416);
    const float mean_vals[3] = {0.f, 0.f, 0.f};
    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
//    const float mean_vals[3] = {103.53f, 116.28f, 123.675f};
//    const float norm_vals[3] = {0.017429f, 0.017507f, 0.017125f};
    int stride = 4;
    int cls_num = 80;
    static char class_names[80];
};



#endif //PPOCR_PPOCR_DET_H
