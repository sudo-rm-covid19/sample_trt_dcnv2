#ifndef MODULATED_DEFORM_CONV_PLUGIN_H
#define MODULATED_DEFORM_CONV_PLUGIN_H

#include "NvInferPlugin.h"
#include "serialize.hpp"
#include "plugin.h"
#include <string>
#include <vector>
#include "modulatedDeformConv.h"

using namespace nvinfer1::plugin;
namespace nvinfer1
{
namespace plugin
{

class ModulatedDeformConvPlugin : public IPluginV2DynamicExt
{
public:
	ModulatedDeformConvPlugin(const int stride, const int padding, const int dilation, const int with_bias,
					 const Dims weight_size, const Dims bias_size, const int deformable_groups, const int groups, 
					 const Weights& weight, const Weights& bias);

	ModulatedDeformConvPlugin(const int stride, const int padding, const int dilation, const int with_bias,
					 const Dims weight_size, const Dims bias_size, const int deformable_groups, const int groups, 
					 const std::vector<float>& weight_value, const std::vector<float>& bias_value, const int weight_vol,
					 const int bias_vol, const DataType type);
	
	ModulatedDeformConvPlugin(void const* serialData, size_t serialLength);

	ModulatedDeformConvPlugin() = delete;
	~ModulatedDeformConvPlugin() override;

	IPluginV2DynamicExt* clone () const override;
	
	DimsExprs getOutputDimensions (int outputIndex, const DimsExprs *inputs, 
		int nbInputs, IExprBuilder &exprBuilder) override;

	bool supportsFormatCombination (int pos, const PluginTensorDesc *inOut, 
		int nbInputs, int nbOutputs) override;

	void configurePlugin (const DynamicPluginTensorDesc *in, int nbInputs, 
		const DynamicPluginTensorDesc *out, int nbOutputs) override;

	size_t getWorkspaceSize (const PluginTensorDesc *inputs, int nbInputs, 
		const PluginTensorDesc *outputs, int nbOutputs) const override;

	int enqueue (const PluginTensorDesc *inputDesc, const PluginTensorDesc *outputDesc, 
		const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

	nvinfer1::DataType getOutputDataType (int index, const nvinfer1::DataType *inputTypes, 
		int nbInputs) const override;

	void attachToContext (cudnnContext *cudnnContext, cublasContext *cublasContext, IGpuAllocator *gpuAllocator) override;

	void detachFromContext () override;

	const char* getPluginType () const override;

	const char*	getPluginVersion () const override;

	int getNbOutputs () const override;

	int	initialize () override;

	void terminate () override;

	size_t getSerializationSize () const override;

	void serialize (void *buffer) const override;

	void destroy () override;

	void setPluginNamespace (const char *pluginNamespace) override;

	const char*	getPluginNamespace () const override;

private:
	int mDilation;
	int mPadding;
	int mStride;
	int mWithBias;	
	int mDeformableGroups;
	int mGroups;
	Dims mWeightSize;
	Dims mBiasSize;

	std::vector<long> mWeightSizeVec;
	std::vector<long> mBiasSizeVec;
	int mWeightVol;
	int mBiasVol;

	Weights mWeight;
	Weights mBias;
	std::vector<float> mWeightValuesHost;
	std::vector<float> mBiasValuesHost;
	float* mWeightValuesDevice;
	float* mBiasValuesDevice;
	at::Tensor mWeightTensorDevice;
	at::Tensor mBiasTensorDevice;
	bool mInitialized;

	DataType mType;
	at::TensorOptions mTensorOptions;

	const char* mPluginNamespace;
};

class ModulatedDeformConvPluginCreator : public BaseCreator
{
public:
	ModulatedDeformConvPluginCreator();

	~ModulatedDeformConvPluginCreator() override = default;

	const char* getPluginName () const override;

	const char* getPluginVersion () const override;

	const PluginFieldCollection* getFieldNames () override;

	IPluginV2* createPlugin (const char *name, const PluginFieldCollection *fc) override;

	IPluginV2* deserializePlugin (const char *name, const void *serialData, size_t serialLength) override;
	
private:
    static PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
};

}	//namespace plugin
}	//namespace nvinfer1
#endif	//MODULATED_DEFORM_CONV_PLUGIN_H
