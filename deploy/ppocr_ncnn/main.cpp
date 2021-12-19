#include <iostream>
#include "ppocr_det.h"


int main() {
    cv::Mat img;
    std::string path2 = "/home/tjm/Documents/python/pycharmProjects/PaddleOCR/samples/ger_1.jpg";
    img = cv::imread(path2);
    if (img.empty()){
        fprintf(stderr, "cv::imread %s failed!", path2.c_str());
        return -1;
    }
    OCRTextDet det("/home/tjm/Documents/python/pycharmProjects/PaddleOCR/onnx/ncnn/ppocr_det.param",
                   "/home/tjm/Documents/python/pycharmProjects/PaddleOCR/onnx/ncnn/ppocr_det.bin");
    det.detector(img);
    cv::imshow("res", img);
    cv::waitKey(0);

    return 0;
}
