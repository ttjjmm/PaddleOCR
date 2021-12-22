#include <iostream>
#include "ppocr_det.h"
#include "ppocr_cls.h"

//using namespace cv;


int main() {


    // /home/tjm/Documents/python/pycharmProjects
    // /home/ubuntu/Documents/pycharm/
    cv::Mat img;
    std::string path2 = "/home/tjm/Documents/python/pycharmProjects/PaddleOCR/samples/11.jpg";
    img = cv::imread(path2);
    if (img.empty()){
        fprintf(stderr, "cv::imread %s failed!", path2.c_str());
        return -1;
    }

//    ResizeImg(img, resize_img, 960, rat_a, rat_b);

    OCRTextDet det("/home/tjm/Documents/python/pycharmProjects/PaddleOCR/onnx/ncnn/ppocr_det.param",
                   "/home/tjm/Documents/python/pycharmProjects/PaddleOCR/onnx/ncnn/ppocr_det.bin");
    auto res = det.detector(img);

    for (auto& det_box: res){
        cv::rectangle(img, det_box.pt1, det_box.pt2, cv::Scalar(255, 0, 0), 2, cv::LINE_4);
    }
    cv::imshow("res", img);
    cv::waitKey(0);

    return 0;
}
