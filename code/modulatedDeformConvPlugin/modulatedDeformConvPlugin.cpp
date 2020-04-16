#include "modulatedDeformConvPlugin.h"
#include <iostream>
#include <cstring>

using namespace nvinfer1;
using namespace modulated_deform_conv;
using nvinfer1::plugin::ModulatedDeformConvPlugin;
using nvinfer1::plugin::ModulatedDeformConvPluginCreator;

namespace modulated_deform_conv
{
// Helper functions
inline void printDim(const Dims* dims, const char* var_name)
{
    for(int i=0; i<dims->nbDims; i++)
    {
        printf("%s Dim[%d]: %d \n", var_name, i, dims->d[i]);
    }
}

inline void printDimExpr(const DimsExprs* dims, const char* var_name)
{
    for(int i=0; i<dims->nbDims; i++)
    {
        printf("%s Dim[%d]: %d \n", var_name, i, dims->d[i]->getConstantValue());
    }
}

inline std::vector<long> dim2vec(const Dims* dims)
{
	int nbElement = static_cast<int> (dims->nbDims);
	std::vector<long> out(nbElement);
	for(int i=0; i<nbElement; i++)
	{
		out[i] = static_cast<long> (dims->d[i]);
		// std::cout << out[i] << std::endl;
	}
	return out;
}

inline Dims makeDims(int nbDims, int val)
{
    nvinfer1::Dims dims;
    dims.nbDims = nbDims;
    std::fill_n(dims.d, nbDims, val);
    return dims;
}

inline Dims vec2dim(const std::vector<long> vec)
{
	int nbElement = static_cast<int> (vec.size());
	Dims out = makeDims(nbElement, 0);
	for (int i=0; i<nbElement; i++)
	{
		out.d[i] = static_cast<int> (vec[i]);
	}
	return out;
}

inline size_t calVolume(const Dims* dims)
{
	int nbElement = static_cast<int> (dims->nbDims);
	size_t volume = 1;
	for (int i=0; i<nbElement; i++)
	{
		volume = volume * static_cast<size_t> (dims->d[i]);
	}
	return volume;
}	
}
namespace
{

const char* MODULATED_DEFORM_CONV_PLUGIN_VERSION{"1"};
const char* MODULATED_DEFORM_CONV_PLUGIN_NAME{"ModulatedDeformConv"};
const char* MODULATED_DEFORM_CONV_PLUGIN_NAMESPACE{""};
}

// Static class fields initialization
PluginFieldCollection ModulatedDeformConvPluginCreator::mFC{};
std::vector<PluginField> ModulatedDeformConvPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ModulatedDeformConvPluginCreator);

ModulatedDeformConvPlugin::ModulatedDeformConvPlugin(const int stride, const int padding, const int dilation, const int with_bias,
					 const Dims weight_size, const Dims bias_size, const int deformable_groups, const int groups, 
					 const Weights& weight, const Weights& bias)
	: mStride(stride), mPadding(padding), mDilation(dilation), mWithBias(with_bias),
	mWeightSize(weight_size), mBiasSize(bias_size), mDeformableGroups(deformable_groups),
	mGroups(groups), mWeight(weight), mBias(bias), mInitialized(false)
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin constructor(createPlugin)." << std::endl;
#endif

	ASSERT(weight.type == nvinfer1::DataType::kFLOAT && "Only support float now"); 
	ASSERT(bias.type == nvinfer1::DataType::kFLOAT && "Only support float now");
    
    mWeightVol = static_cast<int> (weight.count);
    mBiasVol = static_cast<int> (bias.count);

    mWeightValuesHost.assign((float*) weight.values, (float*) weight.values + weight.count);  
    mBiasValuesHost.assign((float*) bias.values, (float*) bias.values + bias.count); 

    mWeightSizeVec = dim2vec(&weight_size);
	mBiasSizeVec = dim2vec(&bias_size);
   
    mType = weight.type;
    mTensorOptions = mTensorOptions.device(c10::kCUDA);
	mTensorOptions = mTensorOptions.dtype(c10::kFloat);
}

// Constructor for clone (with bias) method. we don't use weights since the data may have been corrupted
ModulatedDeformConvPlugin::ModulatedDeformConvPlugin(const int stride, const int padding, const int dilation, const int with_bias,
							 const Dims weight_size, const Dims bias_size, const int deformable_groups, const int groups, 
							 const std::vector<float>& weight_value, const std::vector<float>& bias_value, const int weight_vol,
							 const int bias_vol, const DataType type)
	: mStride(stride), mPadding(padding), mDilation(dilation), mWithBias(with_bias),
	mWeightSize(weight_size), mBiasSize(bias_size), mDeformableGroups(deformable_groups),
	mGroups(groups), mWeightValuesHost(weight_value), mBiasValuesHost(bias_value),
	mWeightVol(weight_vol), mBiasVol(bias_vol), mType(type), mInitialized(false)
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin constructor(clone)." << std::endl;
#endif

	mWeightSizeVec = dim2vec(&weight_size);
	mBiasSizeVec = dim2vec(&bias_size);

	ASSERT(mType == nvinfer1::DataType::kFLOAT && "Only support float now"); 
    
    mTensorOptions = mTensorOptions.device(c10::kCUDA);
	mTensorOptions = mTensorOptions.dtype(c10::kFloat);

#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Exit ModulatedDeformConvPlugin constructor(clone)." << std::endl;
#endif
}

ModulatedDeformConvPlugin::ModulatedDeformConvPlugin(void const* serialData, size_t serialLength)
	:mInitialized(false)
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin constructor(deserialize)." << std::endl;
#endif	
	deserialize_value(&serialData, &serialLength, &mStride);
	deserialize_value(&serialData, &serialLength, &mPadding);
	deserialize_value(&serialData, &serialLength, &mDilation);
	deserialize_value(&serialData, &serialLength, &mWithBias);
	deserialize_value(&serialData, &serialLength, &mWeightSizeVec);
	deserialize_value(&serialData, &serialLength, &mBiasSizeVec);
	deserialize_value(&serialData, &serialLength, &mDeformableGroups);
	deserialize_value(&serialData, &serialLength, &mGroups);
	deserialize_value(&serialData, &serialLength, &mWeightValuesHost);
	deserialize_value(&serialData, &serialLength, &mBiasValuesHost);
	deserialize_value(&serialData, &serialLength, &mWeightVol);
	deserialize_value(&serialData, &serialLength, &mBiasVol);
	deserialize_value(&serialData, &serialLength, &mType);

#if MODULATED_DEFORM_CONV_DEBUG_DATA
	std::cout << "data in deserialize constructor!" << std::endl;
	std::cout << "stride: " << mStride << std::endl;
	std::cout << "padding: " << mPadding << std::endl;
	std::cout << "dilation: " << mDilation << std::endl;
	std::cout << "with_bias: " << mWithBias << std::endl;
	std::cout << "deformable_groups: " << mDeformableGroups << std::endl;
	std::cout << "groups: " << mGroups << std::endl;
	std::cout << "weight_vol: " << mWeightVol << std::endl;
	std::cout << "bias_vol: " << mBiasVol << std::endl;
	std::cout << "mWeightValuesHost count: " << mWeightValuesHost.size() << std::endl;
	std::cout << "mBiasValuesHost count: " << mBiasValuesHost.size() << std::endl;
#endif

	mWeightSize = vec2dim(mWeightSizeVec);
	mBiasSize = vec2dim(mBiasSizeVec);

#if MODULATED_DEFORM_CONV_DEBUG_DATA
	printDim(&mBiasSize, "mBiasSize");
	printDim(&mWeightSize, "mWeightSize");
#endif

	ASSERT(mType == nvinfer1::DataType::kFLOAT && "Only support float now"); 
    mTensorOptions = mTensorOptions.device(c10::kCUDA);
	mTensorOptions = mTensorOptions.dtype(c10::kFloat);
}

ModulatedDeformConvPlugin::~ModulatedDeformConvPlugin()
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin destructor." << std::endl;
#endif
	terminate();
}

void ModulatedDeformConvPlugin::configurePlugin(const DynamicPluginTensorDesc *in, int nbInputs, 
		const DynamicPluginTensorDesc *out, int nbOutputs)
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin::configurePlugin" << std::endl;
#endif
	ASSERT(nbInputs == 3 && "input number must be 3");
	ASSERT(nbOutputs == 1 && "output number must be 1");

	ASSERT(in[0].desc.dims.nbDims == 4);
	ASSERT(in[1].desc.dims.nbDims == 4);
	ASSERT(in[2].desc.dims.nbDims == 4);
	ASSERT(in[0].desc.type == in[1].desc.type && "input data type has to be identical");
	ASSERT(in[1].desc.type == in[2].desc.type && "input data type has to be identical");

	ASSERT(in[0].desc.format == TensorFormat::kLINEAR &&
		   in[1].desc.format == TensorFormat::kLINEAR && 
		   in[2].desc.format == TensorFormat::kLINEAR &&
		   out[0].desc.format == TensorFormat::kLINEAR);

	for (int i = 0; i < nbInputs; i++)
    {
      for (int j = 0; j < in[0].desc.dims.nbDims; j++)
      {
        // Do not support dynamic dimensions
        ASSERT(in[i].desc.dims.d[j] != -1);
      }
    }

#if MODULATED_DEFORM_CONV_DEBUG_DATA
	std::cout << "dilation: " << mDilation << std::endl;
	std::cout << "padding: " << mPadding << std::endl;
	std::cout << "stride: " << mStride << std::endl;
	std::cout << "with bias: " << mWithBias << std::endl;
	std::cout << "deformable_groups: " << mDeformableGroups << std::endl;
	std::cout << "groups: " << mGroups << std::endl;
	printDim(&mWeightSize, "weight size");
	printDim(&mBiasSize, "bias size");		
#endif

#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Exit ModulatedDeformConvPlugin::configurePlugin" << std::endl;
#endif
}

bool ModulatedDeformConvPlugin::supportsFormatCombination (int pos, const PluginTensorDesc *inOut, 
		int nbInputs, int nbOutputs)
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin::supportsFormatCombination" << std::endl;	
#endif

	ASSERT(nbInputs == 3 && nbOutputs == 1 && pos < nbInputs + nbOutputs);
	bool condition = inOut[pos].format == TensorFormat::kLINEAR;
	condition &= inOut[pos].type == DataType::kFLOAT;

#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Exit ModulatedDeformConvPlugin::supportsFormatCombination" << std::endl;
#endif
	return condition;
}

nvinfer1::DataType ModulatedDeformConvPlugin::getOutputDataType (int index, const nvinfer1::DataType *inputTypes, 
		int nbInputs) const
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin::getOutputDataType" << std::endl;
#endif
	ASSERT(index == 0);
	ASSERT(inputTypes[0] == inputTypes[1] && "DataType has to be identical");
	ASSERT(inputTypes[1] == inputTypes[2] && "DataType has to be identical");

	ASSERT(inputTypes[0] == DataType::kFLOAT && "currently only support float32 datatype")

#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Exit ModulatedDeformConvPlugin::getOutputDataType" << std::endl;
#endif

	return inputTypes[0];	
}

IPluginV2DynamicExt* ModulatedDeformConvPlugin::clone () const
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin::clone" << std::endl;
#endif
	
	auto plugin = new ModulatedDeformConvPlugin(mStride, mPadding, mDilation, mWithBias,
			mWeightSize, mBiasSize, mDeformableGroups, mGroups, mWeightValuesHost, 
			mBiasValuesHost, mWeightVol, mBiasVol, mType);

	plugin->setPluginNamespace(mPluginNamespace);

#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Exit ModulatedDeformConvPlugin::clone" << std::endl;
#endif
	return plugin;
}

void ModulatedDeformConvPlugin::attachToContext (cudnnContext *cudnnContext, 
	cublasContext *cublasContext, IGpuAllocator *gpuAllocator) {}

void ModulatedDeformConvPlugin::detachFromContext () {}

const char* ModulatedDeformConvPlugin::getPluginType () const
{
	return MODULATED_DEFORM_CONV_PLUGIN_NAME;
}

const char*	ModulatedDeformConvPlugin::getPluginVersion () const
{
	return MODULATED_DEFORM_CONV_PLUGIN_VERSION;
}

int ModulatedDeformConvPlugin::getNbOutputs () const
{
	return 1;
}

DimsExprs ModulatedDeformConvPlugin::getOutputDimensions (int outputIndex, const DimsExprs *inputs, 
		int nbInputs, IExprBuilder &exprBuilder)
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin::getOutputDimensions" << std::endl;
#endif

	DimsExprs output(inputs[0]);

	int channels_out = mWeightSize.d[0];
	ASSERT(inputs[0].d[2]->isConstant() && "input dimension is not build time constant");
	ASSERT(inputs[0].d[3]->isConstant() && "input dimension is not build time constant");
	int height = inputs[0].d[2]->getConstantValue();
	int width = inputs[0].d[3]->getConstantValue();
	int kernel_h = mWeightSize.d[2];
	int kernel_w = mWeightSize.d[3];

	int height_out = (height + 2 * mPadding - (mDilation * (kernel_h - 1) + 1)) / mStride + 1;
	int width_out = (width + 2 * mPadding - (mDilation * (kernel_w - 1) + 1)) / mStride + 1;

	const IDimensionExpr* C = exprBuilder.constant(channels_out);
	const IDimensionExpr* H = exprBuilder.constant(height_out);
	const IDimensionExpr* W = exprBuilder.constant(width_out);
	output.d[1] = C;	
	output.d[2] = H;	
	output.d[3] = W;

#if MODULATED_DEFORM_CONV_DEBUG_DATA
	printDimExpr(&output, "output");
#endif

#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Exit ModulatedDeformConvPlugin::getOutputDimensions" << std::endl;
#endif
	// ASSERT(false);
	return output;
}

int	ModulatedDeformConvPlugin::initialize ()
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin::initialize" << std::endl;	
#endif
	const size_t weight_bytes = mWeightVol * sizeof(float);	
	const size_t bias_bytes = mBiasVol * sizeof(float);

#if MODULATED_DEFORM_CONV_DEBUG_DATA
	std::cout << "\nweight count: " << mWeightVol << std::endl;
	std::cout << "bias count: " << mBiasVol << std::endl;
#endif 

	CHECK_PLUGIN_STATUS(cudaMalloc((void**) &mBiasValuesDevice, bias_bytes));
	CHECK_PLUGIN_STATUS(cudaMemcpy(mBiasValuesDevice, mBiasValuesHost.data(), bias_bytes, cudaMemcpyHostToDevice));
	mBiasTensorDevice = at::from_blob((void*) mBiasValuesDevice, mBiasSizeVec, [](void*){}, mTensorOptions);
	
	CHECK_PLUGIN_STATUS(cudaMalloc((void**) &mWeightValuesDevice, weight_bytes));
	CHECK_PLUGIN_STATUS(cudaMemcpy(mWeightValuesDevice, mWeightValuesHost.data(), weight_bytes, cudaMemcpyHostToDevice));
	mWeightTensorDevice = at::from_blob((void*) mWeightValuesDevice, mWeightSizeVec, [](void*){}, mTensorOptions);

	mInitialized = true;

	return STATUS_SUCCESS;	
}

void ModulatedDeformConvPlugin::terminate () 
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
    std::cout << "Calling ModulatedDeformConvPlugin::terminate. " << std::endl;
#endif
    if (mInitialized)
    {
    	mInitialized = false;

		CHECK_PLUGIN_STATUS(cudaFree(mWeightValuesDevice));
		CHECK_PLUGIN_STATUS(cudaFree(mBiasValuesDevice));		
    }   

#if MODULATED_DEFORM_CONV_DEBUG_FUNC
    std::cout << "Exit ModulatedDeformConvPlugin::terminate. " << std::endl;
#endif
}

size_t ModulatedDeformConvPlugin::getWorkspaceSize (const PluginTensorDesc *inputs, int nbInputs, 
		const PluginTensorDesc *outputs, int nbOutputs) const
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin::getWorkspaceSize" << std::endl;
#endif

	return get_workspace_size(inputs[0].dims, mWeightSize.d[2], mPadding, mPadding, 
                         mWeightSize.d[3], mStride, mStride, mDilation, mDilation);
}

int ModulatedDeformConvPlugin::enqueue (const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, 
		const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPlugin::enqueue" << std::endl;
#endif

#if MODULATED_DEFORM_CONV_DEBUG_TIME
	GpuTimer timer;
#endif

#if MODULATED_DEFORM_CONV_DEBUG_DATA
	printDim(&inputDesc[0].dims, "input[0]_dims: ");
	printDim(&inputDesc[1].dims, "input[1]_dims: ");
	printDim(&inputDesc[2].dims, "input[2]_dims: ");
	printDim(&outputDesc[0].dims, "output_dims: ");
#endif	

	ASSERT(mInitialized && "Weight tensor not initialized, check initialize().")
	
#if MODULATED_DEFORM_CONV_DEBUG_DATA
	const size_t weight_bytes = mWeightVol * sizeof(float);
	const size_t bias_bytes = mBiasVol * sizeof(float);
	float* weight_host = (float*) malloc(weight_bytes);
	float* bias_host = (float*) malloc(bias_bytes);
	memset(weight_host, 0, weight_bytes);
	memset(bias_host, 0, bias_bytes);
	CHECK_PLUGIN_STATUS(cudaMemcpy(weight_host, (void*)mWeightValuesDevice, weight_bytes, cudaMemcpyDeviceToHost));
	CHECK_PLUGIN_STATUS(cudaMemcpy(bias_host, (void*)mBiasValuesDevice, bias_bytes, cudaMemcpyDeviceToHost));
	
	at::TensorOptions tensorOpt_cpu;
	tensorOpt_cpu = tensorOpt_cpu.device(c10::kCPU);
	tensorOpt_cpu = tensorOpt_cpu.dtype(c10::kFloat);
	at::Tensor weight_tensor_cpu = at::from_blob((void*) weight_host, mWeightSizeVec, [](void*){}, tensorOpt_cpu);
	at::Tensor bias_tensor_cpu = at::from_blob((void*) bias_host, mBiasSizeVec, [](void*){}, tensorOpt_cpu);
	// std::cout << "weight tensor: " << std::endl;
	// at::print(std::cout, weight_tensor_cpu, 99);
	// std::cout << "\nbias tensor: " << std::endl;
	// at::print(std::cout, bias_tensor_cpu, 99);
	free(weight_host);
	free(bias_host);
#endif 

	std::vector<long> input_sizes = dim2vec(&inputDesc[0].dims);
	std::vector<long> offset_sizes = dim2vec(&inputDesc[1].dims);
	std::vector<long> mask_sizes = dim2vec(&inputDesc[2].dims);
	std::vector<long> output_sizes = dim2vec(&outputDesc[0].dims);

#if MODULATED_DEFORM_CONV_DEBUG_TIME
	timer.Start();
#endif
	at::Tensor input_gpu = at::from_blob((void*) inputs[0], input_sizes, [](void*){}, mTensorOptions);
	at::Tensor offset_gpu = at::from_blob((void*) inputs[1], offset_sizes, [](void*){}, mTensorOptions);
	at::Tensor mask_gpu = at::from_blob((void*) inputs[2], mask_sizes, [](void*){}, mTensorOptions);
	at::Tensor output_gpu = at::from_blob((void*) outputs[0], output_sizes, [](void*){}, mTensorOptions);

#if MODULATED_DEFORM_CONV_DEBUG_TIME
	timer.Stop();
	printf("Create tensor form gpu time: %f \n", timer.Elapsed());
#endif

	int status = -1;

#if MODULATED_DEFORM_CONV_DEBUG_TIME
	timer.Start();
#endif
	status = modulated_deform_conv_cuda(input_gpu, mWeightTensorDevice, mBiasTensorDevice, offset_gpu,
                     		  mask_gpu, output_gpu, mWeightSize.d[2], mWeightSize.d[3], mStride, mStride, 
                     		  mPadding, mPadding, mDilation, mDilation, mGroups, mDeformableGroups,
                     		  mWithBias, stream, workspace, mTensorOptions);
	ASSERT(status == 1 && "Status is not SUCCESS!");

#if MODULATED_DEFORM_CONV_DEBUG_TIME
	timer.Stop();
	std::cout << "Time in modulated_deform_conv_cuda: " << timer.Elapsed() << std::endl;
#endif
	// ASSERT(false);
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "\nExit ModulatedDeformConvPlugin::enqueue" << std::endl;
#endif
	return 0;
}

size_t ModulatedDeformConvPlugin::getSerializationSize () const
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "\nCalling ModulatedDeformConvPlugin::getSerializationSize" << std::endl;
#endif
	
	return (serialized_size(mStride) +
			serialized_size(mPadding) +
			serialized_size(mDilation) +
			serialized_size(mWithBias) +
			serialized_size(mWeightSizeVec) +
			serialized_size(mBiasSizeVec) +
			serialized_size(mDeformableGroups) +
			serialized_size(mGroups) + 
			serialized_size(mWeightValuesHost) +
			serialized_size(mBiasValuesHost) +
			serialized_size(mWeightVol) + 
			serialized_size(mBiasVol) +
			serialized_size(mType));
}

void ModulatedDeformConvPlugin::serialize (void *buffer) const
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "\nCalling ModulatedDeformConvPlugin::serialize" << std::endl;
#endif
	serialize_value(&buffer, mStride);
	serialize_value(&buffer, mPadding);
	serialize_value(&buffer, mDilation);
	serialize_value(&buffer, mWithBias);
	serialize_value(&buffer, mWeightSizeVec);
	serialize_value(&buffer, mBiasSizeVec);
	serialize_value(&buffer, mDeformableGroups);
	serialize_value(&buffer, mGroups);
	serialize_value(&buffer, mWeightValuesHost);
	serialize_value(&buffer, mBiasValuesHost);
	serialize_value(&buffer, mWeightVol);
	serialize_value(&buffer, mBiasVol);
	serialize_value(&buffer, mType);
}

void ModulatedDeformConvPlugin::destroy ()
{
	delete this;
}

void ModulatedDeformConvPlugin::setPluginNamespace (const char *pluginNamespace)
{
	mPluginNamespace = pluginNamespace;
}

const char*	ModulatedDeformConvPlugin::getPluginNamespace () const
{
	return mPluginNamespace;
}

ModulatedDeformConvPluginCreator::ModulatedDeformConvPluginCreator()
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPluginCreator constructor!" << std::endl;
#endif
	setPluginNamespace(MODULATED_DEFORM_CONV_PLUGIN_NAMESPACE);

	mPluginAttributes.emplace_back(PluginField("stride", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("padding", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("dilation", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("groups", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("deformable_groups", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("with_bias", nullptr, PluginFieldType::kINT32, 1));
	mPluginAttributes.emplace_back(PluginField("weight_size", nullptr, PluginFieldType::kDIMS, 1));
	mPluginAttributes.emplace_back(PluginField("weight_value", nullptr, PluginFieldType::kFLOAT32, 1));
	mPluginAttributes.emplace_back(PluginField("bias_size", nullptr, PluginFieldType::kDIMS, 1));
	mPluginAttributes.emplace_back(PluginField("bias_value", nullptr, PluginFieldType::kFLOAT32, 1));

	mFC.nbFields = mPluginAttributes.size();
	mFC.fields = mPluginAttributes.data();

#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Exit ModulatedDeformConvPluginCreator constructor!" << std::endl;
#endif
}

const char* ModulatedDeformConvPluginCreator::getPluginName () const
{
	return MODULATED_DEFORM_CONV_PLUGIN_NAME;
}

const char* ModulatedDeformConvPluginCreator::getPluginVersion () const
{
	return MODULATED_DEFORM_CONV_PLUGIN_VERSION;
}

const PluginFieldCollection* ModulatedDeformConvPluginCreator::getFieldNames ()
{
	return &mFC;
}

IPluginV2* ModulatedDeformConvPluginCreator::createPlugin (const char *name, const PluginFieldCollection *fc)
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPluginCreator::createPlugin!" << std::endl;
#endif

	const PluginField* fields = fc->fields;
	
	int nbFields = fc->nbFields;
	
	int stride;
	int padding;
	int dilation;
	int groups;
	int deformable_groups;
	int with_bias;

	Dims weight_size;
	std::vector<float> weight_value;
	Dims bias_size;
	std::vector<float> bias_value;

	for (int i=0; i<nbFields; i++)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "with_bias"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			with_bias = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
	}

	if (!with_bias)
	{
		bias_size = makeDims(1, 0);
		bias_value.push_back(0.0);
	}

	for (int i=0; i<nbFields; i++)
	{
		const char* attrName = fields[i].name;
		if (!strcmp(attrName, "stride"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			stride = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "padding"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			padding = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "dilation"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			dilation = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "groups"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			groups = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "deformable_groups"))
		{
			ASSERT(fields[i].type == PluginFieldType::kINT32);
			deformable_groups = static_cast<int>(*(static_cast<const int*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "weight_size"))
		{
			ASSERT(fields[i].type == PluginFieldType::kDIMS);
			weight_size = static_cast<Dims>(*(static_cast<const Dims*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "weight_value"))
		{
			ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
			int size = fields[i].length;
			weight_value.reserve(size);
			const auto* w = static_cast<const float*>(fields[i].data);
			for (int j=0; j<size; j++)
			{
				// std::cout << *w << std::endl;
				weight_value.push_back(*w);
				w++;
			}
		}
		else if (!strcmp(attrName, "bias_size") && with_bias)
		{
			ASSERT(fields[i].type == PluginFieldType::kDIMS);
			bias_size = static_cast<Dims>(*(static_cast<const Dims*>(fields[i].data)));
		}
		else if (!strcmp(attrName, "bias_value") && with_bias)
		{
			ASSERT(fields[i].type == PluginFieldType::kFLOAT32);
			int size = fields[i].length;
			bias_value.reserve(size);
			const auto* w = static_cast<const float*>(fields[i].data);
			for (int j=0; j<size; j++)
			{
				// std::cout << *w << std::endl;
				bias_value.push_back(*w);
				w++;
			}
		}
	}

#if MODULATED_DEFORM_CONV_DEBUG_DATA
	std::cout << "with_bias: " << with_bias << std::endl;
	std::cout << "stride: " << stride << std::endl;
	std::cout << "padding: " << padding << std::endl;
	std::cout << "dilation: " << dilation << std::endl;
	std::cout << "groups: " << groups << std::endl;
	std::cout << "deformable_groups: " << deformable_groups << std::endl;
	printDim(&weight_size, "weight_size in creator");
	std::cout << "weight length: " << weight_value.size() << std::endl;
	if (with_bias)
	{
		printDim(&bias_size, "bias_size in creator");
		std::cout << "bias length: " << bias_value.size() << std::endl;
	}
	
#endif

	Weights weight{DataType::kFLOAT, weight_value.data(), (int64_t) weight_value.size()};	
	Weights bias{DataType::kFLOAT, bias_value.data(), (int64_t) bias_value.size()};

	auto plugin = new ModulatedDeformConvPlugin(stride, padding, dilation, with_bias,
				 weight_size, bias_size, deformable_groups, groups, weight, bias);

	plugin->setPluginNamespace(mNamespace.c_str());

#if MODULATED_DEFORM_CONV_DEBUG_FUNC
		std::cout << "Exit ModulatedDeformConvPluginCreator::createPlugin!" << std::endl;
#endif
	return plugin;
}

IPluginV2* ModulatedDeformConvPluginCreator::deserializePlugin (const char *name, const void *serialData, size_t serialLength)
{
#if MODULATED_DEFORM_CONV_DEBUG_FUNC
	std::cout << "Calling ModulatedDeformConvPluginCreator::deserializePlugin!" << std::endl;
#endif
	ModulatedDeformConvPlugin* plugin = new ModulatedDeformConvPlugin(serialData, serialLength);
	plugin->setPluginNamespace(mNamespace.c_str());
	return plugin;
}