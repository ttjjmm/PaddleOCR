//
// Created by tjm on 2021/12/18.
//
#include <algorithm>
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
        in = ncnn::Mat::from_pixels(dst.data, ncnn::Mat::PIXEL_BGR, dst_w, dst_h);
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
//    std::cout << width << "x" << height << std::endl;
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
    else {
        for(int i = 0; i < outsize.height; ++i){
            memcpy(dst.data + (i * outsize.width + pad_size.width) * 3,
                   src.data + i * dst_size.width * 3,
                   dst_size.width * 3);
        }
    }//std::cout << "Error Occor!" << std::endl;
    return dst;
}


cv::Mat OCRTextDet::detector(const cv::Mat& image) {
    ncnn::Mat input, preds_map;
    cv::Mat dst;
    dst = preprocess(image, input, false, true);
    cv::imshow("eee", dst);

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
    memcpy((float *)thresh_map.data, preds_map.data, preds_map.h * preds_map.w * sizeof(float));
    this->postprocess(thresh_map);
//    thresh_map.convertTo(thresh_map, CV_8UC1, 255);
//    cv::imshow("r", thresh_map);
//    std::cout << thresh_map;
//    std::cout << preds_map.w << "x" << preds_map.h << "x" << preds_map.c << std::endl;
//    CenterDet::draw_bboxes(dst, dets);
    return thresh_map;
}


float box_score_fast(const cv::Mat& bitmap, const cv::Mat& boxPts) {
    int w = bitmap.cols;
    int h = bitmap.rows;

    double min, max;
    int xmin, xmax;
    int ymin, ymax;

    cv::minMaxLoc(boxPts.col(0), &min, &max);
    xmin = (std::max)((int)floor(min), 0);
    xmax = (std::min)((int)ceil(max), w - 1);

    cv::minMaxLoc(boxPts.col(1), &min, &max);
    ymin = (std::max)((int)floor(min), 0);
    ymax = (std::min)((int)ceil(max), h - 1);

    cv::Mat mask = cv::Mat::zeros(cv::Size(xmax - xmin + 1, ymax - ymin + 1), CV_8UC1);

    cv::Mat new_pts = boxPts.clone();
    new_pts.col(0) = new_pts.col(0) - xmin;
    new_pts.col(1) = new_pts.col(1) - ymin;
    // TODO type
    new_pts.convertTo(new_pts, CV_32SC1, 1, -0.5);
//    std::vector<std::vector<cv::Point2f>> pts_array;
    std::vector<cv::Mat> pts;
    pts.push_back(new_pts);

    cv::fillPoly(mask, pts, 1);
    cv::Scalar score = cv::mean(bitmap(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)), mask);
//    std::cout << score[0] << std::endl;
    return (float)score[0];
}


cv::Mat get_min_box(const std::vector<cv::Point>& points){
    cv::RotatedRect x = cv::minAreaRect(points);
    cv::Mat boxPts;
    cv::boxPoints(x, boxPts);

    cv::sort(boxPts, boxPts, cv::SORT_EVERY_COLUMN);

    int inds[4];
//    uchar index_1, index_2, index_3, index_4;
    if(boxPts.at<float>(1, 1) > boxPts.at<float>(0, 1)){
        inds[0] = 0; inds[3] = 1;
    } else{
        inds[0] = 1; inds[3] = 0;
    }
    if(boxPts.at<float>(3, 1) > boxPts.at<float>(2, 1)){
        inds[1] = 2; inds[2] = 3;
    } else{
        inds[1] = 3; inds[2] = 2;
    }

    cv::Mat new_boxpts = cv::Mat::zeros(cv::Size(2, 4), CV_32FC1);

    for(auto i = 0; i < 4; ++i){
        cv::Mat temp_pt = boxPts.row(inds[i]);
        for (auto j = 0; j < 2; ++j){
            new_boxpts.at<float>(i, j) = temp_pt.at<float>(0, j);
        }
    }
    return new_boxpts;
}

// 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


void OCRTextDet::postprocess(const cv::Mat& src){
    cv::Mat seg = src > this->thresh;
    seg.convertTo(seg, CV_8UC1, 255);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(seg, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<float> scores;
    std::vector<cv::Mat> pts;

    for(auto &contour : contours){
        auto pt = get_min_box(contour);
        std::cout << pt << std::endl;
        pts.push_back(pt);
        scores.push_back(box_score_fast(src, pt));
    }

    for(auto &s: scores){
        std::cout << s << std::endl;
    }

//    box_score_fast(seg, s);
//    std::cout << contours.size() << std::endl;

//    for(auto i = contours.begin(); i != contours.end(); ++i){
//        std::cout << *i << std::endl;
//    }
//    std::cout << contours.size();
    cv::imshow("rs", seg);
//    std::cout << seg;
}



