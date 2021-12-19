#include <iostream>
#include "ppocr_det.h"


int main() {
    cv::Mat img;
    img = cv::imread("/home/tjm/Documents/python/pycharmProjects/PaddleOCR/samples/ger_1.jpg");
    cv::imshow("res", img);
    cv::waitKey(0);

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
