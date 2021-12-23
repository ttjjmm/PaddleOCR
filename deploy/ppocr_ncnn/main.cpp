#include <iostream>
#include "include/ppocr_det.h"
#include "include/ppocr_cls.h"
#include "include/ppocr_rec.h"

//using namespace cv;


int main() {


    // /home/tjm/Documents/python/pycharmProjects
    // /home/ubuntu/Documents/pycharm/
    cv::Mat img;
    std::string path2 = "/home/ubuntu/Documents/pycharm/PaddleOCR/samples/word_1.png";
    img = cv::imread(path2);
    if (img.empty()){
        fprintf(stderr, "cv::imread %s failed!", path2.c_str());
        return -1;
    }

//    ResizeImg(img, resize_img, 960, rat_a, rat_b);

    OCRTextRec det("/home/ubuntu/Documents/pycharm/PaddleOCR/onnx/ncnn/pdocrv2.0_rec-op.param",
                   "/home/ubuntu/Documents/pycharm/PaddleOCR/onnx/ncnn/pdocrv2.0_rec-op.bin",
                   "/home/ubuntu/Documents/pycharm/PaddleOCR/utils/paddleocr_keys.txt");
    det.detector(img);

//    for (auto& det_box: res){
//        cv::rectangle(img, det_box.pt1, det_box.pt2, cv::Scalar(255, 0, 0), 2, cv::LINE_4);
//    }
    cv::imshow("res", img);
    cv::waitKey(0);

    return 0;
}
