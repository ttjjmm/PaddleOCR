//
// Created by ubuntu on 2021/12/20.
//

#include "include/ppocr_rec.h"
#include "benchmark.h"

OCRTextRec::OCRTextRec(const char *param, const char *bin, const char *label_path) {
    this->net = new ncnn::Net();
    this->net->opt.use_fp16_arithmetic = true;
//    this->net->opt.use_vulkan_compute = true;
    this->net->load_param(param);
    this->net->load_model(bin);
    this->label_list = OCRTextRec::read_dict(label_path);
}


OCRTextRec::~OCRTextRec() {
    delete this->net;
}


void OCRTextRec::resize(const cv::Mat &img, cv::Mat &resize_img, float &ratio_w, float &ratio_h) const {

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


cv::Mat OCRTextRec::resize(const cv::Mat& image, const cv::Size_<int>& outsize) {

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


void pretty_print(const ncnn::Mat& m)
{
    for (int q=0; q<m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y=0; y<m.h; y++)
        {
            for (int x=0; x<m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

template <class ForwardIterator>
size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

void OCRTextRec::detector(const cv::Mat &image) {
    ncnn::Mat in, preds_map;
    cv::Mat resize_img;
    float ratio_w, ratio_h;
    std::cout << image.cols << " x " << image.rows << std::endl;
    resize_img = this->resize(image, this->in_size);
//    cv::imshow("vis", resize_img);
//    cv::waitKey(0);
    std::cout << resize_img.cols << " x " << resize_img.rows << std::endl;
    in = ncnn::Mat::from_pixels(resize_img.data,
                                ncnn::Mat::PIXEL_RGB,
                                resize_img.cols,
                                resize_img.rows);

    in.substract_mean_normalize(this->mean_vals, this->norm_vals);
//    dst = preprocess(image, input, false, true);
    ncnn::Mat temp;
    double start = ncnn::get_current_time();
    auto ex = this->net->create_extractor();
//    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input("input", in);
    ex.extract("out", preds_map);

    double end = ncnn::get_current_time();
    printf("Cost Time:%7.2f\n", end - start);

    auto* floatArray = (float*)preds_map.data;
    std::vector<float> outputData(floatArray, floatArray + preds_map.h * preds_map.w);


    std::cout << "dim: " << preds_map.dims << ", w:" << preds_map.w << ", h:" << preds_map.h << ", c:" << preds_map.c << std::endl;
//    pretty_print(preds_map);
//    exit(11);
    int maxIndex;
    float maxValue;
    std::vector<RecInfo> labels;
    for (auto i = 0; i < preds_map.h; ++i) {
        maxIndex = 0;
        maxValue = -1000.f;

        maxIndex = int(argmax(outputData.begin() + i * preds_map.w,
                              outputData.begin() + i * preds_map.w + preds_map.w));
        maxValue = float(*std::max_element(outputData.begin() + i * preds_map.w, outputData.begin() + i * preds_map.w + preds_map.w));// / partition;
//        std::cout << preds_map.w << std::endl;

//        if (maxIndex > 0 && maxIndex < keySize && (!(i > 0 && maxIndex == lastIndex))) {
//            scores.emplace_back(maxValue);
//            strRes.append(keys[maxIndex - 1]);
//        }
//        lastIndex = maxIndex;
//        std::cout << max_score << std::endl;

        std::cout << maxIndex << " , " << this->label_list[maxIndex] << std::endl;
        labels.push_back(RecInfo{maxValue, maxIndex});
    }


}


std::vector<std::string> OCRTextRec::decode(const std::vector<RecInfo> &pred_labels) {

}


std::vector<std::string> OCRTextRec::read_dict(const std::string& path){
    std::ifstream in(path);
    std::string line;
    std::vector<std::string> m_vec;
    if(in) {
        while (getline(in, line)) {
            m_vec.push_back(line);
        }
    } else {
        std::cout << "no such lable file: " << path << ", exit the program..." << std::endl;
        exit(1);
    }
    return m_vec;
}







