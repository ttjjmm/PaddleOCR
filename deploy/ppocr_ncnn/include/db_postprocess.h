//
// Created by tjm on 2021/12/22.
//

#ifndef PPOCR_DB_POSTPROCESS_H
#define PPOCR_DB_POSTPROCESS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include "include/clipper.cpp"

class DBPostProcess {

public:

    std::vector<cv::Point> unclip(const cv::Mat& box, const float &unclip_ratio);
    float box_score_fast(const cv::Mat& bitmap, const cv::Mat& boxPts);
    cv::Mat get_min_box(const std::vector<cv::Point>& points, float& ssid);
    void mat2points(const cv::Mat& src, std::vector<cv::Point2i>& pt_vec, const int& w, const int& h, const float& ratio_w, const float& ratio_h);

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
