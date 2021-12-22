//
// Created by tjm on 2021/12/18.
//
#include <iostream>
#include <algorithm>
#include "ppocr_det.h"
#include "benchmark.h"
#include "include/clipper.cpp"

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
    else {
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
    if (ratio != 1.0){
        dst_size = cv::Size(floor(width * 1.0 / ratio), floor(height * 1.0 / ratio));
        cv::resize(image, src, dst_size, cv::INTER_LINEAR);
        std::cout << dst_size.height << " x " << dst_size.width << std::endl;
    } else{
        dst_size = cv::Size(width, height);
    }
    pad_size = cv::Size(floor((outsize.width - dst_size.width) / 2), floor((outsize.height - dst_size.height) / 2));
    std::cout << "into height" << pad_size << std::endl;

    if (pad_size.height != 0){
        // inital pointer of dst mat
        auto init = dst.data + 3 * outsize.width * pad_size.height;
        for (int i = 0; i < dst_size.height; ++i){
            memcpy(init + i * outsize.width * 3,
                   src.data + i * outsize.width * 3,
                   outsize.width * 3);
        }
    }
    else if (pad_size.width != 0){
        for (int i = 0; i < outsize.height; ++i){
            memcpy(dst.data + (i * outsize.width + pad_size.width) * 3,
                   src.data + i * dst_size.width * 3,
                   dst_size.width * 3);
        }
    }
    else {
        for (int i = 0; i < outsize.height; ++i){
            memcpy(dst.data + (i * outsize.width + pad_size.width) * 3,
                   src.data + i * dst_size.width * 3,
                   dst_size.width * 3);
        }
    }//std::cout << "Error Occor!" << std::endl;
    return dst;
}


void OCRTextDet::resize(const cv::Mat &img, cv::Mat &resize_img, float &ratio_w, float &ratio_h) const {

    int w = img.cols;
    int h = img.rows;
    float ratio = 1.f;
    int max_wh = w > h ? w: h;
    if (max_wh > this->max_side_len) {
        if (h > w) {
            ratio = float(this->max_side_len) / float(h);
        } else {
            ratio = float(this->max_side_len) / float(w);
        }
    }
    int resize_w = int(float(w) * ratio);
    int resize_h = int(float(h) * ratio);
    resize_w = std::max(int(round(float(resize_w) / 32) * 32), 32);
    resize_h = std::max(int(round(float(resize_h) / 32) * 32), 32);

    cv::resize(img, resize_img, cv::Size(resize_w, resize_h));
    ratio_w = float(resize_w) / float(w);
    ratio_h = float(resize_h) / float(h);
}


std::vector<BoxInfo> OCRTextDet::detector(const cv::Mat &image) {
    ncnn::Mat in, preds_map;
    cv::Mat resize_img;
    float ratio_w, ratio_h;
    this->resize(image, resize_img, ratio_w, ratio_h);
//    cv::imshow("eee", resize_img);
//    resize_img.convertTo(resize_img, CV_32FC3, 1. / 255);
    in = ncnn::Mat::from_pixels(resize_img.data,
                                   ncnn::Mat::PIXEL_BGR,
                                   resize_img.cols,
                                   resize_img.rows);

    in.substract_mean_normalize(this->mean_vals, this->norm_vals);
//    dst = preprocess(image, input, false, true);


    double start = ncnn::get_current_time();
    auto ex = this->net->create_extractor();
//    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input("input.1", in);
    ex.extract("preds", preds_map);

    double end = ncnn::get_current_time();

    printf("Cost Time:%7.2f\n", end - start);
    cv::Mat binary_map(cv::Size(preds_map.w, preds_map.h), CV_32FC1);
    memcpy((float *)binary_map.data, preds_map.data, preds_map.h * preds_map.w * sizeof(float));

    std::vector<BoxInfo> boxes;
    this->postprocess(binary_map, boxes, ratio_w, ratio_h);
    return boxes;
}



void GetContourArea(const cv::Mat& box,
                    float unclip_ratio, float& distance) {
    int pts_num = 4;
    float area = 0.0f;
    float dist = 0.0f;
    for (int i = 0; i < pts_num; i++) {
        area += box.at<float>(i, 0) * box.at<float>((i + 1) % pts_num, 1) -
                box.at<float>(i, 1) * box.at<float>((i + 1) % pts_num, 0);
        dist += sqrtf(
                (box.at<float>(i, 0) - box.at<float>((i + 1) % pts_num, 0)) *
                (box.at<float>(i, 0) - box.at<float>((i + 1) % pts_num, 0)) +
                (box.at<float>(i, 1) - box.at<float>((i + 1) % pts_num, 1)) *
                (box.at<float>(i, 1) - box.at<float>((i + 1) % pts_num, 1)));
    }
    area = fabs(float(area / 2.0));
    distance = area * unclip_ratio / dist;
}


std::vector<std::vector<float>> mat2vec(const cv::Mat& src) {
    std::vector<std::vector<float>> vec;
    std::vector<float> temp;
    for (auto i = 0; i < src.rows; ++i) {
        temp.clear();
        for (auto j = 0; j < src.cols; ++j) {
            temp.push_back(src.at<float>(i, j));
        }
        vec.push_back(temp);
    }
    return vec;
}


std::vector<cv::Point> unclip(const cv::Mat& box, const float &unclip_ratio) {
    float distance = 1.0;
    GetContourArea(box, unclip_ratio, distance);
    ClipperLib::ClipperOffset offset;
    ClipperLib::Path p;
    p << ClipperLib::IntPoint(int(box.at<float>(0, 0)), int(box.at<float>(0, 1)))
      << ClipperLib::IntPoint(int(box.at<float>(1, 0)), int(box.at<float>(1, 1)))
      << ClipperLib::IntPoint(int(box.at<float>(2, 0)), int(box.at<float>(2, 1)))
      << ClipperLib::IntPoint(int(box.at<float>(3, 0)), int(box.at<float>(3, 1)));
    offset.AddPath(p, ClipperLib::jtRound, ClipperLib::etClosedPolygon);

    ClipperLib::Paths soln;
    offset.Execute(soln, distance);
    std::vector<cv::Point> points;

    for (int j = 0; j < soln.size(); j++) {
        for (int i = 0; i < soln[soln.size() - 1].size(); i++) {
            points.emplace_back(soln[j][i].X, soln[j][i].Y);
        }
    }
    return points;
}


float box_score_fast(const cv::Mat& bitmap, const cv::Mat& boxPts) {
    int w = bitmap.cols;
    int h = bitmap.rows;

    double min, max;
    int xmin, xmax;
    int ymin, ymax;

    cv::minMaxLoc(boxPts.col(0), &min, &max);
    xmin = clamp((int)floor(min), 0, w - 1);
    xmax = clamp((int)ceil(max), 0, w - 1);

    cv::minMaxLoc(boxPts.col(1), &min, &max);
    ymin = clamp((int)floor(min), 0, h - 1);
    ymax = clamp((int)ceil(max), 0, h - 1);

    cv::Mat mask = cv::Mat::zeros(cv::Size(xmax - xmin + 1, ymax - ymin + 1), CV_8UC1);

    cv::Mat new_pts = boxPts.clone();
    new_pts.col(0) -= xmin;
    new_pts.col(1) -= ymin;
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


cv::Mat get_min_box(const std::vector<cv::Point>& points, float& ssid){
    cv::RotatedRect rbox;
    if (points.empty())
        rbox = cv::RotatedRect(cv::Point(0, 0), cv::Size(1, 1), 0);
    else
        rbox = cv::minAreaRect(points);

    ssid = std::max(rbox.size.width, rbox.size.height);

    cv::Mat boxPts;
    cv::boxPoints(rbox, boxPts);

    cv::sort(boxPts, boxPts, cv::SORT_EVERY_COLUMN);
    cv::Mat new_boxpts = cv::Mat::zeros(cv::Size(2, 4), CV_32FC1);

    if (boxPts.at<float>(1, 1) > boxPts.at<float>(0, 1)) {
        boxPts.row(0).copyTo(new_boxpts.row(0));
        boxPts.row(1).copyTo(new_boxpts.row(3));
    } else {
        boxPts.row(1).copyTo(new_boxpts.row(0));
        boxPts.row(0).copyTo(new_boxpts.row(3));
    }
    if (boxPts.at<float>(3, 1) > boxPts.at<float>(2, 1)) {
        boxPts.row(2).copyTo(new_boxpts.row(1));
        boxPts.row(3).copyTo(new_boxpts.row(2));
    } else {
        boxPts.row(3).copyTo(new_boxpts.row(1));
        boxPts.row(2).copyTo(new_boxpts.row(2));
    }
    return new_boxpts;
}

// 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


void mat2points(const cv::Mat& src,
                std::vector<cv::Point2i>& pt_vec,
                const int& w, const int& h,
                const float& ratio_w, const float& ratio_h) {

    double min, max;
    int xmin, xmax, ymin, ymax;

    cv::minMaxLoc(src.col(0), &min, &max);
    xmin = clamp((int)floor(min), 0, w - 1);
    xmax = clamp((int)ceil(max), 0, w - 1);

    min = 0; max = 0;
    cv::minMaxLoc(src.col(1), &min, &max);
    ymin = clamp((int)floor(min), 0, h - 1);
    ymax = clamp((int)ceil(max), 0, h - 1);

    pt_vec.emplace_back(cv::Point(int(float(xmin) / ratio_w), int(float(ymin) / ratio_h)));
    pt_vec.emplace_back(cv::Point(int(float(xmax) / ratio_w), int(float(ymax) / ratio_h)));
}


void OCRTextDet::postprocess(const cv::Mat& src, std::vector<BoxInfo>& boxes,
                             const float& r_w, const float& r_h) const {
    cv::Mat seg = src > this->thresh;
    seg.convertTo(seg, CV_8UC1, 255);
//    cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
//    cv::dilate(seg, seg, dila_ele);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(seg, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    std::vector<float> scores;
    std::vector<cv::Mat> pts;
    const float min_size = 3.;
    const int max_candidates = 1000;
    int num_contours = contours.size() >= max_candidates ? max_candidates : (int)contours.size();
    std::cout << "det before postprocess: " << num_contours << std::endl;
    for(auto i = 0; i < num_contours; ++i) {
        float ssid;
        auto pt = get_min_box(contours[i], ssid);
        if (ssid < min_size) continue;

        float score = box_score_fast(src, pt);
        if (this->box_thresh > score) continue;

        auto x = unclip(pt, this->unclip_ratio);
//        std::cout << x.center << std::endl;
        auto z = get_min_box(x, ssid);

//        if (ssid < min_size + 2) continue;


        pts.push_back(z);
        scores.push_back(score);
    }

    cv::Mat seg_three;
    std::vector<cv::Mat> s{seg * 255, seg *255, seg * 255};
    cv::merge(s, seg_three);
    int w = seg.cols;
    int h = seg.rows;
    std::cout << "det after postprocess: " << pts.size() << std::endl;
    for (auto &pt: pts) {
        std::vector<cv::Point2i> points;
        mat2points(pt, points, w, h, r_w, r_h);
        BoxInfo newbox;
        newbox.score = 0.0;
        newbox.pt1 = points[0];
        newbox.pt2 = points[1];
        boxes.push_back(newbox);
//        std::cout << "x1: " << points[0] << ", x2: " << points[1] << std::endl;
//        cv::rectangle(seg_three, points[0], points[1], cv::Scalar(0, 0, 255), 2, cv::LINE_4);
    }

//    std::cout << seg << std::endl;
    cv::imshow("rs", seg_three);
//    std::cout << seg;
}



