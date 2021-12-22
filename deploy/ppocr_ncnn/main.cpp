#include <iostream>
#include "ppocr_det.h"
#include "ppocr_cls.h"

//using namespace cv;


int main() {

    // /home/tjm/Documents/python/pycharmProjects
    // /home/ubuntu/Documents/pycharm/
    cv::Mat img;
    std::string path2 = "/home/ubuntu/Documents/pycharm/PaddleOCR/samples/00009282.jpg";
    img = cv::imread(path2);
    if (img.empty()){
        fprintf(stderr, "cv::imread %s failed!", path2.c_str());
        return -1;
    }

//    ResizeImg(img, resize_img, 960, rat_a, rat_b);

    OCRTextDet det("/home/ubuntu/Documents/pycharm/PaddleOCR/onnx/ncnn/ppocr_det.param",
                   "/home/ubuntu/Documents/pycharm/PaddleOCR/onnx/ncnn/ppocr_det.bin");
    det.detector(img);

//    cv::imshow("res", img);
    cv::waitKey(0);

    return 0;
}
