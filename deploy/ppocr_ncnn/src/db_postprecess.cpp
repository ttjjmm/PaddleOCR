//
// Created by tjm on 2021/12/22.
//

#include "include/db_postprocess.h"


std::vector<cv::Point> DBPostProcess::unclip(const cv::Mat& box, const float &unclip_ratio) {
    float distance = 1.0;
    DBPostProcess::get_contour_area(box, unclip_ratio, distance);
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


float DBPostProcess::box_score_fast(const cv::Mat& bitmap, const cv::Mat& boxPts) {
    int w = bitmap.cols;
    int h = bitmap.rows;

    double min, max;
    int xmin, xmax;
    int ymin, ymax;

    cv::minMaxLoc(boxPts.col(0), &min, &max);
    xmin = this->clamp((int)floor(min), 0, w - 1);
    xmax = this->clamp((int)ceil(max), 0, w - 1);

    cv::minMaxLoc(boxPts.col(1), &min, &max);
    ymin = this->clamp((int)floor(min), 0, h - 1);
    ymax = this->clamp((int)ceil(max), 0, h - 1);

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


cv::Mat DBPostProcess::get_min_box(const std::vector<cv::Point>& points, float& ssid){
    cv::RotatedRect rbox;
    if (points.empty())
        rbox = cv::RotatedRect(cv::Point(0, 0), cv::Size(1, 1), 0);
    else
        rbox = cv::minAreaRect(points);

    ssid = std::max(rbox.size.width, rbox.size.height);

    cv::Mat boxPts, sort_idx;
    cv::Mat sorted_boxpts = cv::Mat::zeros(cv::Size(2, 4), CV_32FC1);
    cv::boxPoints(rbox, boxPts);

    cv::sortIdx(boxPts.col(0), sort_idx, cv::SORT_EVERY_COLUMN);

    for (auto i = 0; i < 4; ++i) {
        boxPts.row(sort_idx.at<int>(i)).copyTo(sorted_boxpts.row(i));
    }

    cv::Mat new_boxpts = cv::Mat::zeros(cv::Size(2, 4), CV_32FC1);

    if (sorted_boxpts.at<float>(1, 1) > sorted_boxpts.at<float>(0, 1)) {
        sorted_boxpts.row(0).copyTo(new_boxpts.row(0));
        sorted_boxpts.row(1).copyTo(new_boxpts.row(3));
    } else {
        sorted_boxpts.row(1).copyTo(new_boxpts.row(0));
        sorted_boxpts.row(0).copyTo(new_boxpts.row(3));
    }
    if (sorted_boxpts.at<float>(3, 1) > sorted_boxpts.at<float>(2, 1)) {
        sorted_boxpts.row(2).copyTo(new_boxpts.row(1));
        sorted_boxpts.row(3).copyTo(new_boxpts.row(2));
    } else {
        sorted_boxpts.row(3).copyTo(new_boxpts.row(1));
        sorted_boxpts.row(2).copyTo(new_boxpts.row(2));
    }
    return new_boxpts;
}


void DBPostProcess::get_contour_area(const cv::Mat& box, float unclip_ratio, float& distance) {
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


void DBPostProcess::mat2points(const cv::Mat& src,
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


void DBPostProcess::box_from_bitmap(const cv::Mat& src, std::vector<BoxInfo>& boxes, const float& thresh,
                                    const float& box_thresh, const float& unclip_ratio,
                                    const float& r_w, const float& r_h) {
    cv::Mat seg = src > thresh;
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
        if (box_thresh > score) continue;

        auto x = unclip(pt, unclip_ratio);
//        std::cout << x.center << std::endl;
        auto z = get_min_box(x, ssid);

        if (ssid < min_size + 2) continue;


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





