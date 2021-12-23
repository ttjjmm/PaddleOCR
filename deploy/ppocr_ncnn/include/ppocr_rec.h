//
// Created by ubuntu on 2021/12/20.
//

#ifndef PPOCR_PPOCR_REC_H
#define PPOCR_PPOCR_REC_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"


typedef struct RecInfo{
    float score;
    int index;
} RecInfo;


class OCRTextRec{

public:
    OCRTextRec(const char* param, const char* bin, const char* label_path);
    ~OCRTextRec();
    ncnn::Net* net{};
    void detector(const cv::Mat &image);
//    friend void display(CenterDet *cdt);

private:
    static std::vector<std::string> read_dict(const std::string& path);
    std::vector<std::string> decode(const std::vector<RecInfo> &pred_labels);
    static cv::Mat resize(const cv::Mat& image, const cv::Size_<int>& outsize);
//    cv::Mat preprocess(const cv::Mat& image, ncnn::Mat& in, bool org_size=false, bool keep_ratio=false);
    void resize(const cv::Mat &img, cv::Mat &resize_img, float &ratio_w, float &ratio_h) const;
//    static cv::Mat resize(const cv::Mat& image, const cv::Size_<int>& outsize);
//    void postprocess(const cv::Mat& src, std::vector<BoxInfo>& boxes, const float& r_w, const float& r_h) const;
    std::vector<std::string> label_list;
    int max_side_len = 720;
    cv::Size_<int> in_size = cv::Size(320, 32);
//    const float mean_vals[3] = {0.f, 0.f, 0.f};;
//    const float norm_vals[3] = {1/255.f, 1/255.f, 1/255.f};
    const float mean_vals[3] = {127.5f, 127.5f, 127.5f};;
    const float norm_vals[3] = {1/127.5f, 1/127.5f, 1/127.5f};
};







#endif //PPOCR_PPOCR_REC_H
