#!/bin/bash
set -ex
CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/usr/local/Ascend"}
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

echo "1. install akernel and openyuanrong"
pip install akernel_sdk-0.9.19-py3-none-any.whl && pip install openyuanrong_sdk==0.7.47

echo "2. install vllm-ascend from source"
git clone -b releases/v0.23.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
export COMPILE_CUSTOM_KERNELS=1
pip install -v -e . --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn 
cd ..

echo "3. install mooncake"
git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
apt-get install mpich libmpich-dev -y
bash dependencies.sh -y
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON -DPython3_EXECUTABLE="$(which python)"
make -j
make install
export LD_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
cd ..

echo "4.install verl"
git clone https://github.com/verl-project/verl.git
cd verl && git checkout v0.9.0
pip install -r requirements-npu.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
pip install -v -e .
cd ..

echo "7.apply patch"
cd vllm-ascend
git apply --whitespace=nowarn ../verl-ascend-recipe/pd_disaggregation/patch/vllm-ascend.patch && cd ..

cd verl
git apply --whitespace=nowarn ../verl-ascend-recipe/pd_disaggregation/patch/verl.patch && cd ..


