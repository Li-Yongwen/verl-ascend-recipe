#!/bin/bash
set -ex

apt-get update
mkdir -p /var/lib/alternatives
apt-get install mpich libmpich-dev -y

CANN_INSTALL_PATH=${CANN_INSTALL_PATH:-"/mnt/share/t00986241/b106"}
source ${CANN_INSTALL_PATH}/ascend-toolkit/set_env.sh
source ${CANN_INSTALL_PATH}/nnal/atb/set_env.sh

echo "1. install akernel and openyuanrong"
pip install akernel_sdk-0.9.19-py3-none-any.whl

echo "2. install vllm-ascend from source"
git clone -b releases/v0.23.0 https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && pip install -r requirements.txt --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn
export COMPILE_CUSTOM_KERNELS=1
MAX_JOBS=128 pip install -v -e . --extra-index-url https://triton-ascend.osinfra.cn/pypi/simple/ --trusted-host triton-ascend.osinfra.cn 
cd ..

echo "3. install mooncake"
git clone -b v0.3.9 --depth 1 https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
echo 'check_certificate = off' >> /etc/wgetrc
sed -i 's|https://go.dev/dl/|https://golang.google.cn/dl/|g' dependencies.sh
sed -i '249s#golang\.google\.cn/dl#mirrors.aliyun.com/golang#g' dependencies.sh
bash dependencies.sh -y
mkdir build
cd build
cmake .. -DUSE_ASCEND_DIRECT=ON -DPython3_EXECUTABLE="$(which python)"
make -j128
make install
cp mooncake-common/src/libmooncake_common.so ${CANN_INSTALL_PATH}/ascend-toolkit/latest/python/site-packages/mooncake
cp mooncake-transfer-engine/src/libtransfer_engine.so ${CANN_INSTALL_PATH}/ascend-toolkit/latest/python/site-packages/mooncake
cp mooncake-store/src/libmooncake_store.so ${CANN_INSTALL_PATH}/ascend-toolkit/latest/python/site-packages/mooncake
export LD_LIBRARY_PATH=${CANN_INSTALL_PATH}/ascend-toolkit/latest/python/site-packages/mooncake:$LD_LIBRARY_PATH
cd ../../

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

echo "5.run uni-agent"
git clone https://github.com/chifeichi/uni-agent.git
cd uni-agent && git checkout cfc/new_release
bash examples/blackbox_recipes/claude_code/run_train_no_pd.sh
