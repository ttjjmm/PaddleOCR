//
// Created by ubuntu on 2021/12/20.
//

#ifndef PPOCR_PPOCR_BASE_H
#define PPOCR_PPOCR_BASE_H


#include <iostream>
#include <opencv2/opencv.hpp>
#include "net.h"


class PPOCRBase{

public:
    PPOCRBase(const char* param, const char* bin);
    ~PPOCRBase(){
        delete this->net;
    }

    virtual cv::Mat inference(const cv::Mat& image){}

    ncnn::Net* net{};

private:
    cv::Mat preprocess(const cv::Mat& image, ncnn::Mat& in, bool org_size=false, bool keep_ratio=false);
//    static cv::Mat resize(const cv::Mat& image, const cv::Size_<int>& outsize);
    virtual void postprocess(const cv::Mat& src){}

    cv::Size_<int> in_size = cv::Size(640, 640);
//    const float mean_vals[3] = {0.f, 0.f, 0.f};
//    const float norm_vals[3] = {1/255.f, 1/255.f, 1/225.f};
    const float mean_vals[3] = {103.94f, 116.78f, 123.68f};
    const float norm_vals[3] = {0.017f, 0.017f, 0.017f};
    float thresh = 0.3;
};


//class PPOCRDetector: public PPOCRBase{
//
//public:
//    // use base class initialize the parameters
//    PPOCRDetector(const char* param, const char* bin): PPOCRBase(param, bin){};
//    ~PPOCRDetector(){
//        delete this->net;
//    }
//
//private:
//    void postprocess(const cv::Mat& src) override;
//};



#endif //PPOCR_PPOCR_BASE_H
