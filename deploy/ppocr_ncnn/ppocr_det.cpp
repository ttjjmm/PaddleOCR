//
// Created by tjm on 2021/12/18.
//

#include "ppocr_det.h"
#include "benchmark.h"

OCRTextDet::OCRTextDet(const char *param, const char *bin) {
    this->net = new ncnn::Net();
    this->net->opt.use_fp16_arithmetic = true;
//    this->net->opt.use_vulkan_compute = true;
    this->net->load_param(param);
    this->net->load_model(bin);
}


OCRTextDet::~OCRTextDet() {
    delete this->net;
}


cv::Mat OCRTextDet::preprocess(const cv::Mat &image, ncnn::Mat& in, bool org_size, bool keep_ratio){
    cv::Mat dst;
    int dst_w = 0;
    int dst_h = 0;
    if (!org_size) {
        if (keep_ratio) {
            dst = OCRTextDet::resize(image, this->in_size);
//            cv::imshow("ss", dst);
//            cv::waitKey(0);
        } else cv::resize(image, dst,  this->in_size);
        dst_w = dst.cols;
        dst_h = dst.rows;
        in = ncnn::Mat::from_pixels(dst.data, ncnn::Mat::PIXEL_RGB, dst_w, dst_h);
    }
    else{
        dst_w = image.cols;
        dst_h = image.rows;
        in = ncnn::Mat::from_pixels(image.data, ncnn::Mat::PIXEL_RGB, dst_w, dst_h);
    }
    // convert to nccn's mat
//    in = ncnn::Mat::from_pixels(dst.data, ncnn::Mat::PIXEL_RGB, dst_w, dst_h);
    in.substract_mean_normalize(this->mean_vals, this->norm_vals);
    return dst;
}


cv::Mat OCRTextDet::resize(const cv::Mat& image, const cv::Size_<int>& outsize) {

    assert(image.channels() == 3);

    int width = image.cols;
    int height = image.rows;
    float ratio_w, ratio_h;
    ratio_w = float (width - 0.1) / outsize.width;
    ratio_h = float (height- 0.1) / outsize.height;
    auto ratio = std::max(ratio_h, ratio_w);

    cv::Mat dst(outsize, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Size_<int> dst_size, pad_size;
    cv:: Mat src;
    if(ratio != 1.0){
        dst_size = cv::Size(floor(width * 1.0 / ratio), floor(height * 1.0 / ratio));
        cv::resize(image, src, dst_size, cv::INTER_LINEAR);
        std::cout << dst_size.height << " x " << dst_size.width << std::endl;
    } else{
        dst_size = cv::Size(width, height);
    }
    pad_size = cv::Size(floor((outsize.width - dst_size.width) / 2), floor((outsize.height - dst_size.height) / 2));
    std::cout << "into height" << pad_size << std::endl;

    if(pad_size.height != 0){
        // inital pointer of dst mat
        auto init = dst.data + 3 * outsize.width * pad_size.height;
        for(int i = 0; i < dst_size.height; ++i){
            memcpy(init + i * outsize.width * 3,
                   src.data + i * outsize.width * 3,
                   outsize.width * 3);
        }
    }
    else if(pad_size.width != 0){
        for(int i = 0; i < outsize.height; ++i){
            memcpy(dst.data + (i * outsize.width + pad_size.width) * 3,
                   src.data + i * dst_size.width * 3,
                   dst_size.width * 3);
        }
    }
    else std::cout << "Error Occor!" << std::endl;
    return dst;
}


cv::Mat OCRTextDet::detector(const cv::Mat& image) {
    ncnn::Mat input, preds_map;

    cv::Mat dst;

    dst = preprocess(image, input, false, true);
    std::cout << "xxx:" << input.h << 'x' << input.w << std::endl;
    double start = ncnn::get_current_time();

    auto ex = this->net->create_extractor();
//    ex.set_light_mode(true);
    ex.set_num_threads(4);

    ex.input("input.1", input);
    ex.extract("preds", preds_map);

//    this->decode(heatmap, reg_box, dets, 0.4, 0.6);

    double end = ncnn::get_current_time();
//    std::cout <<  "Inference cost time: " << end - start << std::endl;
    printf("Cost Time:%7.2f\n", end - start);
    cv::Mat thresh_map(preds_map.w, preds_map.h, CV_32FC1);
    memcpy(thresh_map.data, preds_map.data, preds_map.h * preds_map.w * sizeof(float));

    std::cout << preds_map.w << "x" << preds_map.h << "x" << preds_map.c << std::endl;
//    CenterDet::draw_bboxes(dst, dets);
    return thresh_map;
}




