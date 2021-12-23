#!/bin/bash

curr_path=$(pwd)
onnx_path="${curr_path%*scripts}onnx"
# shellcheck disable=SC2082
input_file=$1
file_name="${input_file%*.onnx}"

#echo "${curr_path}"
#echo "${onnx_path}"

cd ~/opt/ncnn-20211208-full-source/build/tools/onnx/
./onnx2ncnn "${onnx_path}/$1" "${onnx_path}/ncnn/${file_name}.param" "${onnx_path}/ncnn/${file_name}.bin"
# optimize tool
cd ..
./ncnnoptimize "${onnx_path}/ncnn/${file_name}.param" "${onnx_path}/ncnn/${file_name}.bin" "${onnx_path}/ncnn/${file_name}.param" "${onnx_path}/ncnn/${file_name}.bin" 65535