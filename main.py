import common
import torch
import numpy as np
import time

feat_size = 32		#32
in_channel = 64		#64
out_channel = 128	#128
kernel_size = 3		#3
batch = 64			#64

feat = torch.randn(
	batch, in_channel, feat_size, feat_size, requires_grad=True, device="cuda")

ONNX_MODEL_NAME = "model.onnx"
ENGINE_NAME = "serialization.engine"

def test_model_export():
	import torch.nn as nn
	import torch.nn.functional as F
	from mmdet.ops import ModulatedDeformConvPack as DeformConv

	class Net(nn.Module):

		def __init__(self):
			super(Net, self).__init__()
			self.deform_conv1 = DeformConv(in_channel, out_channel, kernel_size, padding=1, bias=True)
			self.conv1 = nn.Conv2d(out_channel, out_channel, kernel_size, padding=(1, 1))
			self.deform_conv2 = DeformConv(out_channel, out_channel, kernel_size, padding=1, bias=False)

		def forward(self, x):
			x = self.deform_conv1(x)
			x = F.relu(x)
			x = self.conv1(x)
			x = F.relu(x)
			x = self.deform_conv2(x)
			return x
	model = Net().to("cuda")
	torch.onnx.export(model, feat, ONNX_MODEL_NAME, verbose=True,
					  input_names=['input'], output_names=['output'])
	out = model(feat)
	# for i in range(50):
	# 	start = time.time()
	# 	out = model(feat)
	# 	stop = time.time()
	# 	print ("python inference time: ", stop - start)
	return out.cpu().detach().numpy()

def test_trt_export(model_name=ONNX_MODEL_NAME):
	import tensorrt as trt
	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	trt.init_libnvinfer_plugins(TRT_LOGGER, '')
	
	with trt.Builder(TRT_LOGGER) as builder, builder.create_network(common.EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
		builder.max_workspace_size = common.GiB(1)
		builder.fp16_mode = False
		builder.max_batch_size = 1	

		with open(model_name, 'rb') as model:
		    if not parser.parse(model.read()):
		        print ('ERROR: Failed to parse the ONNX file.')
		        for error in range(parser.num_errors):
		            print (parser.get_error(error))
		        return None

		engine = builder.build_cuda_engine(network)
		print ("CUDA engine build successfully!")
		return engine

def test_serialize_engine(model_name=ONNX_MODEL_NAME, engine_name=ENGINE_NAME):
	with test_trt_export(ONNX_MODEL_NAME) as engine:
		with open(ENGINE_NAME, 'wb') as f:
			f.write(engine.serialize())
		print("engine wrote!")

def test_deserialize_engine(engine_name=ENGINE_NAME):
	import tensorrt as trt
	TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
	trt.init_libnvinfer_plugins(TRT_LOGGER, '')
	with open(engine_name, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
		engine = runtime.deserialize_cuda_engine(f.read())
	print ("load engine!")
	return engine

if __name__ == '__main__':
	# python_output = test_model_export_v2()
	python_output = test_model_export()
	# print(python_output.shape)
	# test_trt_export()
	test_serialize_engine()
	with test_deserialize_engine() as engine:
		inputs, outputs, bindings, stream = common.allocate_buffers(engine, True, context_batch_size=[batch, batch, batch])

		with engine.create_execution_context() as context:		
			# input = np.random.rand(batch, in_channel, feat_size, feat_size).astype('float32')
			# print (input.shape)
			for i in range(50):
				input = feat.cpu().detach().numpy().astype('float32')
				# print ("input[0] python: \n", input)
				start = time.time()
				inputs[0].host = input
				
				trt_outputs = common.do_inference_v2(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
				stop = time.time()
				trt_outputs = np.array(trt_outputs).reshape(-1, out_channel, feat_size, feat_size)
				# print ("output trt: \n", trt_outputs)			

				# print("output python: \n", python_output)
				diff_output = np.mean(python_output - trt_outputs)
				print("output difference trt vs. python: ", diff_output)
				# print("tensorrt inference time: ", stop - start)
