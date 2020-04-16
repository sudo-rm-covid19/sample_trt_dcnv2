#ifndef MODULATED_DEFORM_CONV_H
#define MODULATED_DEFORM_CONV_H

#include <torch/torch.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAEvent.h>
#include <ATen/DeviceGuard.h>
#include "plugin.h"
#include "kernel.h"

namespace modulated_deform_conv
{
#ifndef MODULATED_DEFORM_CONV_DEBUG_FUNC
  #define MODULATED_DEFORM_CONV_DEBUG_FUNC false
#endif

#ifndef MODULATED_DEFORM_CONV_DEBUG_DATA
  #define MODULATED_DEFORM_CONV_DEBUG_DATA false
#endif

#ifndef MODULATED_DEFORM_CONV_DEBUG_TIME
  #define MODULATED_DEFORM_CONV_DEBUG_TIME false
#endif
#ifndef CUDA_MEM_ALIGN 
  #define CUDA_MEM_ALIGN 256
#endif

int modulated_deform_conv_cuda(at::Tensor& input, at::Tensor& weight, at::Tensor& bias, 
                      at::Tensor& offset, at::Tensor& mask, at::Tensor& output, 
                      int kernel_h, int kernel_w, const int stride_h, const int stride_w,
                      const int pad_h, const int pad_w, const int dilation_h,
                      const int dilation_w, const int group, const int deformable_group,
                      const bool with_bias, cudaStream_t& stream, void* workspace,
                      at::TensorOptions tensor_option);

void modulated_deformable_im2col_cuda(const at::Tensor& data_im, const at::Tensor& data_offset,
                    const at::Tensor& data_mask, const int batch_size, const int channels,
                    const int height_im, const int width_im, const int height_col,
                    const int width_col, const int kernel_h, const int kenerl_w,
                    const int pad_h, const int pad_w, const int stride_h, const int stride_w,
                    const int dilation_h, const int dilation_w, const int deformable_group,
                    at::Tensor& data_col, cudaStream_t& stream);

size_t get_workspace_size(const nvinfer1::Dims& input_size, int kernel_h, int pad_h, int pad_w, 
                         int kernel_w, int stride_h, int stride_w, int dilation_h, int dilation_w);

} // namespace modulated_deform_conv
#endif
