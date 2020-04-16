#include "modulatedDeformConv.h"

namespace modulated_deform_conv
{

int modulated_deform_conv_cuda(at::Tensor& input, at::Tensor& weight, at::Tensor& bias, 
                      at::Tensor& offset, at::Tensor& mask, at::Tensor& output, 
                      int kernel_h, int kernel_w, const int stride_h, const int stride_w,
                      const int pad_h, const int pad_w, const int dilation_h,
                      const int dilation_w, const int group, const int deformable_group,
                      const bool with_bias, cudaStream_t& stream, void* workspace,
                      at::TensorOptions tensor_option) {
	AT_CHECK(input.is_contiguous(), "input tensor has to be contiguous");
	AT_CHECK(weight.is_contiguous(), "weight tensor has to be contiguous");
	at::DeviceGuard guard(input.device());

	const int batch = input.size(0);
	const int channels = input.size(1);
	const int height = input.size(2);
	const int width = input.size(3);

	const int channels_out = weight.size(0);
	const int channels_kernel = weight.size(1);
	const int kernel_h_ = weight.size(2);
	const int kernel_w_ = weight.size(3);

	if (kernel_h_ != kernel_h || kernel_w_ != kernel_w)
	AT_ERROR("Input shape and kernel shape wont match: (%d x %d vs %d x %d).",
	         kernel_h_, kernel_w, kernel_h_, kernel_w_);
	if (channels != channels_kernel * group)
	AT_ERROR("Input shape and kernel channels wont match: (%d vs %d).",
	         channels, channels_kernel * group);

	const int height_out =
	  (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int width_out =
	  (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;

	// resize output
	output = output.view({batch, channels_out, height_out, width_out}).zero_();

	// resize temporary columns
	std::vector<long> columns_sizes(2);
	columns_sizes[0] = channels * kernel_h * kernel_w;
	columns_sizes[1] = height_out * width_out;
	size_t columnsWorkSpaceSize = columns_sizes[0] * columns_sizes[1] * sizeof(float);
	void* columnsWorkSpace = static_cast<int8_t*>(workspace);
	at::Tensor columns = at::from_blob(columnsWorkSpace, columns_sizes, [](void*){}, tensor_option);

	output = output.view({output.size(0), group, output.size(1) / group,
	                    output.size(2), output.size(3)});

	for (int b = 0; b < batch; b++) {
		modulated_deformable_im2col_cuda(
		    input[b], offset[b], mask[b], 1, channels, height, width, height_out,
		    width_out, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w,
		    dilation_h, dilation_w, deformable_group, columns, stream);

		// divide into group
		weight = weight.view({group, weight.size(0) / group, weight.size(1),
		                      weight.size(2), weight.size(3)});
		columns = columns.view({group, columns.size(0) / group, columns.size(1)});

		for (int g = 0; g < group; g++) {
		  output[b][g] = output[b][g]
		                     .flatten(1)
		                     .addmm_(weight[g].flatten(1), columns[g])
		                     .view_as(output[b][g]);
		}

		weight = weight.view({weight.size(0) * weight.size(1), weight.size(2),
		                      weight.size(3), weight.size(4)});
		columns =
		    columns.view({columns.size(0) * columns.size(1), columns.size(2)});
	}

	output = output.view({output.size(0), output.size(1) * output.size(2),
	                    output.size(3), output.size(4)});

	if (with_bias) {
		output += bias.view({1, bias.size(0), 1, 1});
	}

	return 1;
}

size_t get_workspace_size(const nvinfer1::Dims& input_size, int kernel_h, int pad_h, int pad_w, 
                         int kernel_w, int stride_h, int stride_w, int dilation_h, int dilation_w)
{
	const int batch = input_size.d[0];
	const int channels = input_size.d[1];
	const int height = input_size.d[2];
	const int width = input_size.d[3];

	const int height_out =
	  (height + 2 * pad_h - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
	const int width_out =
	  (width + 2 * pad_w - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
	
	size_t wss = channels * kernel_h * kernel_w * height_out * width_out * sizeof(float);
	return wss;
}

} // namespace deform_conv
  