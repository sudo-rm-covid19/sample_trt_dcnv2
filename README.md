This repo is to reproduce the error occured in deploying modulated deform conv in centernet,
https://forums.developer.nvidia.com/t/tensorrt-merges-wrongly-two-different-layers/108701/6:

1. add the code in``` /code/builtin_op_imports.cpp```, ```/code/onnx2trt_utils.cpp``` and ```/code/onnx2trt_utils.hpp``` to the corresponding files in TensorRT6 OSS ```/parser/onnx```.

2. files in modulatedDeformConvPlugin is my implementation of dcnv2, I made use of libtorch to implement the enqueue logic, so please include libtorch1.3 in TensorRT OSS project. (To use python inference, build libtorch and pytorch from source is recommended.)

3. to produce the dummy model (which can be successfully run in TensorRT), you need to install mmdetection (https://github.com/open-mmlab/mmdetection) and replace the file in mmdetection ```/mmdet/ops/deform_conv.py``` with the enclosed ```/mmdet/deform_conv.py```. 

4. the error occurs when I run the centernet onnx model provided herein (centernet_dcnv2_new.onnx).

UPDATE:
Our team has found that the problem was due to the slice operation in the implementation of CenterNet and it can be solved by using index_select operation instead.
