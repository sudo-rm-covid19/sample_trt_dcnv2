// Please add this to onnx2trt_utils.hpp
// Helper function to import a plugin from TensorRT's plugin registry given the name and version.
nvinfer1::IPluginV2* importCustomPluginFromRegistry(IImporterContext* ctx, const std::string& pluginName,
    const std::string& pluginVersion, const std::string& nodeName, const std::vector<nvinfer1::PluginField>& pluginFields);
