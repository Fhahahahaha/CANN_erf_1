#!/bin/bash
# Erf算子编译部署与测试脚本
# 使用方法: bash run_test.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "========================================"
echo "Step 1: 编译算子工程"
echo "========================================"
cd "$PROJECT_DIR"
bash build.sh

echo ""
echo "========================================"
echo "Step 2: 部署算子包"
echo "========================================"
./build_out/custom_opp*.run --install-path=${HOME}/

echo ""
echo "========================================"
echo "Step 3: 编译测试程序"
echo "========================================"
g++ -std=c++17 \
    -I${ASCEND_TOOLKIT_HOME}/include \
    -I${HOME}/vendors/customize/op_api/include \
    -L${ASCEND_TOOLKIT_HOME}/lib64 \
    -L${HOME}/vendors/customize/op_api/lib \
    "$SCRIPT_DIR/main.cpp" \
    -lcust_opapi -lnnopbase -lacl_rt \
    -o "$SCRIPT_DIR/execute_erf_op"

echo ""
echo "========================================"
echo "Step 4: 运行测试"
echo "========================================"
source ~/vendors/customize/bin/set_env.bash
./execute_erf_op
