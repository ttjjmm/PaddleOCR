//
// Created by tjm on 2021/12/18.
//

#ifndef PPOCR_PPOCR_DET_H
#define PPOCR_PPOCR_DET_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"
#include "include/db_postprocess.h"
//#include "layer.h"


class OCRTextDet{

public:
    OCRTextDet(const char* param, const char* bin);
    ~OCRTextDet();
    ncnn::Net* net{};
    std::vector<BoxInfo> detector(const cv::Mat &image);
//    friend void display(CenterDet *cdt);

private:
    cv::Mat preprocess(const cv::Mat& image, ncnn::Mat& in, bool org_size=false, bool keep_ratio=false);
    void resize(const cv::Mat &img, cv::Mat &resize_img, float &ratio_w, float &ratio_h) const;
    static cv::Mat resize(const cv::Mat& image, const cv::Size_<int>& outsize);
//    void postprocess(const cv::Mat& src, std::vector<BoxInfo>& boxes, const float& r_w, const float& r_h) const;
    DBPostProcess post_process;
    int max_side_len = 960;
    float thresh = 0.3;
    float box_thresh = 0.5;
    float unclip_ratio = 2;
    cv::Size_<int> in_size = cv::Size(480, 480);

    const float mean_vals[3] = {0.485f*255.f, 0.456f*255.f, 0.406f*255.f};;
    const float norm_vals[3] = {1/0.229f/255.f, 1/0.224f/255.f, 1/0.225f/255.f};
};



#endif //PPOCR_PPOCR_DET_H
