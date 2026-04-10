# DIOYFLLMInfer
自制大模型推理框架，支持CPU和Cuda后端，C++实现。

支持llama2模型。

## 运行环境
运行linux环境ubuntu22

下载模型文件到/models
https://huggingface.co/karpathy/tinyllamas/blob/main/stories110M.bin

## 运行

```shell
mkdir build && cd build
cmake ..
make -j32

# run in protect root path
./build/demo/llama_infer /models/stories110M.bin /models/tokenizer.model

```

# 安装依赖

### Armadillo
```shell
git clone https://gitee.com/mirrors/armadillo-code.git ~/armadillo

cd ~/armadillo
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j8
sudo make install
```

### Google Test

```shell
git clone https://github.com/google/googletest.git ~/googletest

cd ~/googletest
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 
make -j8
sudo make install
```

### Google log

```shell
git clone https://github.com/google/glog.git ~/glog
cd ~/glog
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DWITH_GFLAGS=OFF -DWITH_GTEST=OFF ..
make -j8
sudo make install
```

### SentencePiece（分词库）

```shell
git clone https://github.com/google/sentencepiece.git ~/sentencepiece
cd ~/sentencepiece
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. 
make -j8
sudo make install
```

# 参考

https://github.com/zjhellofss/KuiperLLama