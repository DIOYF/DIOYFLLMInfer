//
// Created by zhou on 2026/4/8.
//

#include "tensor/tensor.h"
#include <glog/logging.h>
#include <numeric>
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <filesystem>


namespace tensor {

    template <typename T, typename Tp>
    static size_t reduce_dimension(T begin, T end, Tp init) {
        if (begin >= end) {
            return 0;
        }
        size_t size = std::accumulate(begin, end, init, std::multiplies<>());
        return size;
    }

    static size_t data_type_size(base::DataType data_type) {
        switch (data_type) {
            case base::DataType::kDataTypeFp32: {
                return 4;
            }
            case base::DataType::kDataTypeInt8: {
                return 1;
            }
            case base::DataType::kDataTypeInt32: {
                return 4;
            }
            default: {
                LOG(FATAL) << "Unknown data_type";
                return 0;
            }
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0,
                    bool need_alloc,std::shared_ptr<base::DeviceAllocator> alloc,
                    void* ptr)
    : data_type_(data_type) {
        dims_.push_back(dim0);
        size_ = dim0;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            if (ptr != nullptr) {
                CHECK(need_alloc == false)
                << "need_alloc is true while ptr is not nullptr";
                init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1,
                    bool need_alloc,std::shared_ptr<base::DeviceAllocator> alloc,
                    void* ptr)
    : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        size_ = dim0 * dim1;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            if (ptr != nullptr) {
                CHECK(need_alloc == false)
                << "need_alloc is true while ptr is not nullptr";
                init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }
    }
    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                    bool need_alloc,std::shared_ptr<base::DeviceAllocator> alloc,
                    void* ptr)
    : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        dims_.push_back(dim2);
        size_ = dim0 * dim1 * dim2;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            if (ptr != nullptr) {
                CHECK(need_alloc == false)
                << "need_alloc is true while ptr is not nullptr";
                init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }
    }

    Tensor::Tensor(base::DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                    bool need_alloc,std::shared_ptr<base::DeviceAllocator> alloc,
                    void* ptr)
    : data_type_(data_type) {
        dims_.push_back(dim0);
        dims_.push_back(dim1);
        dims_.push_back(dim2);
        dims_.push_back(dim3);
        size_ = dim0 * dim1 * dim2 * dim3;
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            if (ptr != nullptr) {
                CHECK(need_alloc == false)
                << "need_alloc is true while ptr is not nullptr";
                init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }
    }

    Tensor::Tensor(base::DataType data_type, std::vector<int32_t> dims,
                    bool need_alloc,std::shared_ptr<base::DeviceAllocator> alloc,
                    void* ptr)
    : data_type_(data_type) {
        dims_ = std::move(dims);
        size_ = reduce_dimension(dims.begin(), dims.end(), 1);
        if (need_alloc && alloc) {
            allocate(alloc);
        } else {
            if (ptr != nullptr) {
                CHECK(need_alloc == false)
                << "need_alloc is true while ptr is not nullptr";
                init_buffer(alloc, data_type_, need_alloc, ptr);
            }
        }
    }

    void Tensor::to_cpu() {
        const base::DeviceType device_type = this->device_type();

        if (device_type == base::DeviceType::kDeviceUnknown) {
            LOG(ERROR) << "Tensor::to_cpu: device_type is unknown";
        } else if (device_type == base::DeviceType::kDeviceCUDA) {
            // todo : move from cuda to cpu
        } else {
            LOG(INFO) << "Tensor::to_cpu: device_type is already cpu";
        }
    }

    void Tensor::to_cuda(cudaStream_t stream) {
        // todo: to_cuda
    }

    bool Tensor::is_empty() const {
        return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
    }

    void Tensor::init_buffer(std::shared_ptr<base::DeviceAllocator> alloc,
        base::DataType data_type,
        bool need_alloc, void* ptr) {
        if (!alloc && !need_alloc) {
            std::shared_ptr<base::Buffer> buffer =
                std::make_shared<base::Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
            this->buffer_ = buffer;
        } else {
            allocate(alloc, true);
        }
    }

    void Tensor::reshape(const std::vector<int32_t>& dims) {
        size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
        if (!buffer_) {
            dims_ = dims;
            size_ = size;
            return;
        }

        if (size > size_) {
            auto new_buffer = std::make_shared<base::Buffer>(data_type_size(data_type_) * size_, buffer_->allocator());
            CHECK(new_buffer->allocate());
            new_buffer->copy_from(buffer_.get());
            this->buffer_ = new_buffer;
        }
        dims_ = dims;
        size_ = size;
    }

    std::shared_ptr<base::Buffer> Tensor::get_buffer() const {return buffer_;}


    size_t Tensor::size() const {return size_;}

    size_t Tensor::byte_size() const { return size_ * data_type_size(data_type_);}

    int32_t Tensor::dim_size() const {return static_cast<int32_t>(dims_.size());}

    base::DataType Tensor::data_type() const {return data_type_;}

    int32_t Tensor::get_dim(int32_t idx) const {
        CHECK_GE(idx , 0);
        CHECK_LT(idx, dims_.size());
        return dims_.at(idx);
    }

    const std::vector<int32_t>& Tensor::dims() const {return dims_;}

    std::vector<size_t> Tensor::strides() const {
        std::vector<size_t> strides;
        if (!dims_.empty()) {
            // 数据连续线性存储，获取数据不同维度的分段stride
            for (int32_t i = 0; i < dims_.size() - 1; ++i) {
                size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
                strides.push_back(stride);
            }
            strides.push_back(1);
        }
        return strides;
    }

    bool Tensor::assign(std::shared_ptr<base::Buffer> buffer) {
        if (!buffer) {
            LOG(ERROR) << "Tensor::assign: buffer is nullptr";
            return false;
        }
        if (buffer_) {
            if (buffer_->device_type() != buffer->device_type()) {
                LOG(ERROR) << "Tensor::assign: buffer device type is different";
                return false;
            }
        }

        size_t byte_size = this->byte_size();
        if (byte_size > buffer->byte_size()) {
            LOG(ERROR) << "The size of buffer is too small";
            return false;
        }
        buffer_ = buffer;
        return true;
    }

    void Tensor::reset(base::DataType data_type, const std::vector<int32_t>& dims) {
        data_type_ = data_type;
        dims_ = dims;
        size_ = reduce_dimension(dims.begin(), dims.end(), 1);
        buffer_ = nullptr;
    }

    void Tensor::set_device_type(base::DeviceType device_type) const {
        if (buffer_) {
            buffer_->set_device_type(device_type);
        }
    }

    base::DeviceType Tensor::device_type() const {
        if (!buffer_){
            return base::DeviceType::kDeviceUnknown;
        }
        return buffer_->device_type();
    }

    bool Tensor::allocate(std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc) {
        if (!allocator) {
            LOG(ERROR) << "Tensor::allocate: allocator is nullptr";
            return false;
        }

        size_t byte_size = this->byte_size();
        if (!byte_size) {
            LOG(ERROR) << "Tensor::allocate: byte_size is zero";
            return false;
        }

        if (buffer_ && byte_size <= buffer_->byte_size()) {
            if (!need_realloc) {
                return true;
            }
        }

        buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
        if (!buffer_->ptr()) {
            LOG(ERROR) << "Tensor::allocate: buffer ptr is nullptr";
            return false;
        }

        return true;
    }

    Tensor Tensor::clone() const {
        Tensor new_tensor = *this;
        size_t byte_size = this->byte_size();

        auto allocator = buffer_->allocator();
        new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
        new_tensor.buffer_->copy_from(buffer_.get());
        return new_tensor;
    }

}

