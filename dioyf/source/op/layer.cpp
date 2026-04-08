//
// Created by zhou on 2026/4/8.
//
#include "op/layer.h"
#include "glog/logging.h"
#include <cstdarg>
#include <numbers>
#include <utility>


namespace op {

    // BaseLayer ==============================================
    BaseLayer::BaseLayer(
        base::DeviceType device_type,
            LayerType layer_type,
            base::DataType data_type,
            std::string layer_name)
    : device_type_(device_type),
    layer_type_(layer_type),
    data_type_(data_type),
    layer_name_(layer_name){}

    BaseLayer::~BaseLayer() {}

    base::DataType BaseLayer::data_type() const { return data_type_; }

    LayerType BaseLayer::layer_type() const { return layer_type_; }

    base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor &weight) {
        return base::error::FunctionNotImplement();
    }

    base::Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
            const void* weight_ptr, base::DeviceType device_type) {
        return base::error::FunctionNotImplement();
    }

    const std::string& BaseLayer::get_layer_name() const { return layer_name_; }

    void BaseLayer::set_layer_name(const std::string& layer_name) { layer_name_ = layer_name; }

    base::DeviceType BaseLayer::device_type() const { return device_type_; }

    void BaseLayer::set_device_type(base::DeviceType device_type) { device_type_ = device_type; }

    // Layer ==============================================
    Layer::Layer(
        base::DeviceType device_type,
        LayerType layer_type,
        std::string layer_name)
    : BaseLayer(device_type, layer_type,
    base::DataType::kDataTypeFp32,
    std::move(layer_name)) {}

    base::Status Layer::init() { return base::error::Success();}

    base::Status Layer::forward() {return base::error::FunctionNotImplement();}

    base::Status Layer::check_tensor(const tensor::Tensor& tensor,
            base::DeviceType device_type,
            base::DataType data_type) const {
        // todo : tensor check

        return base::error::Success();
    }

    base::Status Layer::check_tensor_with_dim(const tensor::Tensor& tensor,
        base::DeviceType device_type,
        base::DataType data_type, ...) const {
        // todo: tensor check
        return base::error::Success();
    }


    base::Status Layer::check() const {
        return base::error::FunctionNotImplement("The check function is not implement yet");
    }

    void Layer::set_input(int32_t idx, const tensor::Tensor& input) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        this->inputs_.at(idx) = input;
    }

    void Layer::set_output(int32_t idx, const tensor::Tensor& output) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        this->outputs_.at(idx) = output;
    }

    tensor::Tensor& Layer::get_input(int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        return inputs_.at(idx);
    }

    tensor::Tensor& Layer::get_output(int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        return outputs_.at(idx);
    }

    const tensor::Tensor& Layer::get_input(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, inputs_.size());
        return inputs_.at(idx);
    }

    const tensor::Tensor& Layer::get_output(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, outputs_.size());
        return outputs_.at(idx);
    }

    size_t Layer::input_size() const {
        return inputs_.size();
    }

    size_t Layer::output_size() const {
        return outputs_.size();
    }

    void Layer::reset_input_size(size_t size) {
        inputs_.resize(size);
    }

    void Layer::reset_output_size(size_t size) {
        outputs_.resize(size);
    }

    void Layer::to_cuda() {
        // todo : cuda support
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& output1) {
        this->set_input(0, input1);
        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& input4,
                                const tensor::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);

        this->set_output(0, output1);
        return this->forward();
    }

    base::Status Layer::forward(const tensor::Tensor& input1, const tensor::Tensor& input2,
                                const tensor::Tensor& input3, const tensor::Tensor& input4,
                                const tensor::Tensor& input5, const tensor::Tensor& output1) {
        this->set_input(0, input1);
        this->set_input(1, input2);
        this->set_input(2, input3);
        this->set_input(3, input4);
        this->set_input(4, input5);

        this->set_output(0, output1);
        return this->forward();
    }


    // LayerParam ==============================================
    LayerParam::LayerParam( base::DeviceType device_type,
            LayerType layer_type,bool is_quant_layer,std::string layer_name
            )
    : Layer(device_type,layer_type, std::move(layer_name)),
    is_quant_layer(is_quant_layer){}


    size_t LayerParam::weight_size() const {
        return weights_.size();
    }

    void LayerParam::reset_weight_size(size_t size) {
        weights_.resize(size);
    }

    tensor::Tensor& LayerParam::get_weight(int32_t idx) {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        return weights_.at(idx);
    }

    const tensor::Tensor& LayerParam::get_weight(int32_t idx) const {
        CHECK_GE(idx, 0);
        CHECK_LT(idx, weights_.size());
        return weights_.at(idx);
    }

    void LayerParam::to_cuda() {
        // todo : to cuda
    }

    base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor& weight) {
        // todo : set weight
        return base::error::Success();
    }

    base::Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims,
        const void* weight_ptr, base::DeviceType device_type) {
        // todo: set dim weight
        return base::error::Success();
    }


    void LayerParam::set_scales(const tensor::Tensor& scales) {
        CHECK(!scales.is_empty());
        this->scales_ = scales;
    }

    void LayerParam::set_group_size(int32_t group_size) {
        this->group_size_ = group_size;
    }

    int32_t LayerParam::get_scale_num() const {
        CHECK(!scales_.is_empty());
        return static_cast<int32_t>(scales_.size());
    }


}
