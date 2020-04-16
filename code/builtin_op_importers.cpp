
// Please add this to builtin_op_importers.cpp
DEFINE_BUILTIN_OP_IMPORTER(ModulatedDeformConv)
{
    std::vector<nvinfer1::ITensor*> tensors;
    tensors.push_back(&convertToTensor(inputs.at(0), ctx)); // input
    tensors.push_back(&convertToTensor(inputs.at(1), ctx)); // offset
    tensors.push_back(&convertToTensor(inputs.at(2), ctx)); // mask

    ASSERT(inputs.at(3).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
    // spatial dimension is restricted to 2 for now.
    nvinfer1::Dims dims = tensors[0]->getDimensions();
    const int nbSpatialDims = dims.nbDims - 2;
    ASSERT(nbSpatialDims == 2, ErrorCode::kUNSUPPORTED_NODE);

    std::vector<nvinfer1::PluginField> f;
    OnnxAttrs attrs(node);

    auto kernel_weight = inputs.at(3).weights();
    nvinfer1::Dims weight_size = kernel_weight.shape;

    int with_bias = attrs.get<int>("with_bias");
    ShapedWeights kernel_bias;
    nvinfer1::Dims bias_size;
    if (with_bias)
    {
        ASSERT(inputs.at(4).is_weights(), ErrorCode::kUNSUPPORTED_NODE);
        kernel_bias = inputs.at(4).weights();
        bias_size = kernel_bias.shape;
    }

    int trt_op_version = attrs.get<int>("trt_op_version");
    const std::string pluginName = "ModulatedDeformConv";
    const std::string pluginVersion = std::to_string(trt_op_version);

    int stride = attrs.get<int>("stride");
    int padding = attrs.get<int>("padding");
    int dilation = attrs.get<int>("dilation");
    int groups = attrs.get<int>("groups");
    int deformable_groups = attrs.get<int>("deformable_groups");

    f.emplace_back("stride", &stride, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("padding", &padding, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("dilation", &dilation, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("groups", &groups, nvinfer1::PluginFieldType::kINT32, 1);    
    f.emplace_back("deformable_groups", &deformable_groups, nvinfer1::PluginFieldType::kINT32, 1);
    f.emplace_back("with_bias", &with_bias, nvinfer1::PluginFieldType::kINT32, 1);

    f.emplace_back("weight_size", &weight_size, nvinfer1::PluginFieldType::kDIMS, 1);
    f.emplace_back("weight_value", kernel_weight.values, nvinfer1::PluginFieldType::kFLOAT32, kernel_weight.count());
    if (with_bias)
    {
        f.emplace_back("bias_size", &bias_size, nvinfer1::PluginFieldType::kDIMS, 1);
        f.emplace_back("bias_value", kernel_bias.values, nvinfer1::PluginFieldType::kFLOAT32, kernel_bias.count());
    }

    nvinfer1::IPluginV2* plugin = importCustomPluginFromRegistry(ctx, pluginName, pluginVersion, node.name(), f);
    ASSERT(plugin != nullptr && "ModulatedDeformConv plugin was not found in the plugin registry!", ErrorCode::kINTERNAL_ERROR);
    
    auto layer = ctx->network()->addPluginV2(tensors.data(), int(tensors.size()), *plugin); 
    ASSERT(layer != nullptr && "Cannot create ModulatedDeformConv plugin!", ErrorCode::kINTERNAL_ERROR);
    layer->setOutputType(0, tensors[0]->getType());

#if ONNX_DEBUG_FUNC
    printf("Exit ModulatedDeformConv importer.\n");
#endif
    // ASSERT(false, ErrorCode::kUNSUPPORTED_GRAPH);
    RETURN_FIRST_OUTPUT(layer);
}