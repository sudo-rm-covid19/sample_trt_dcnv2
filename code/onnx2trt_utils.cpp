
// Please add this to onnx2trt_utils.cpp
nvinfer1::IPluginV2* importCustomPluginFromRegistry(IImporterContext* ctx, const std::string& pluginName,
    const std::string& pluginVersion, const std::string& nodeName, const std::vector<nvinfer1::PluginField>& pluginFields)
{
    const auto mPluginRegistry = getPluginRegistry();

    const auto pluginCreator = mPluginRegistry->getPluginCreator(pluginName.c_str(), pluginVersion.c_str(), "");

    if (!pluginCreator)
    {
      return nullptr;
    }

    nvinfer1::PluginFieldCollection fc;
    fc.nbFields = pluginFields.size();
    fc.fields = pluginFields.data();

    return pluginCreator->createPlugin(nodeName.c_str(), &fc);
}
