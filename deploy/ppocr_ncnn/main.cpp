#include <iostream>
#include "ppocr_det.h"
#include "ppocr_cls.h"

using namespace cv;

int main() {

    cv::Mat img;
    std::string path2 = "/home/ubuntu/Documents/pycharm/PaddleOCR/samples/word_201.png";
    img = cv::imread(path2);
    if (img.empty()){
        fprintf(stderr, "cv::imread %s failed!", path2.c_str());
        return -1;
    }
    OCRTextCls det("/home/ubuntu/Documents/pycharm/PaddleOCR/onnx/ncnn/ppocr_cls.param",
                   "/home/ubuntu/Documents/pycharm/PaddleOCR/onnx/ncnn/ppocr_cls.bin");
    det.detector(img);
    cv::imshow("res", img);
    cv::waitKey(0);

    return 0;
}
