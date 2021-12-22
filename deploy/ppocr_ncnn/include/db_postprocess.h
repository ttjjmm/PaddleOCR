//
// Created by tjm on 2021/12/22.
//

#ifndef PPOCR_DB_POSTPROCESS_H
#define PPOCR_DB_POSTPROCESS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/clipper.h"


typedef struct BoxInfo{
    cv::Point pt1;
    cv::Point pt2;
    float score;
}BoxInfo;


class DBPostProcess {

public:

    static std::vector<cv::Point> unclip(const cv::Mat& box, const float &unclip_ratio);
    float box_score_fast(const cv::Mat& bitmap, const cv::Mat& boxPts);
    static cv::Mat get_min_box(const std::vector<cv::Point>& points, float& ssid);
    void mat2points(const cv::Mat& src, std::vector<cv::Point2i>& pt_vec, const int& w, const int& h, const float& ratio_w, const float& ratio_h);
    void box_from_bitmap(const cv::Mat& src, std::vector<BoxInfo>& boxes, const float& thresh,
                         const float& box_thresh, const float& unclip_ratio,
                         const float& r_w, const float& r_h);
//    std::vector<std::vector<std::vector<int>>> FilterTagDetRes(std::vector<std::vector<std::vector<int>>> boxes, float ratio_h, float ratio_w, cv::Mat srcimg);
private:
    static void get_contour_area(const cv::Mat& box, float unclip_ratio, float& distance);

    template <class T> inline T clamp(T x, T min, T max) {
        if (x > max)
            return max;
        if (x < min)
            return min;
        return x;
    }
};

#endif //PPOCR_DB_POSTPROCESS_H
