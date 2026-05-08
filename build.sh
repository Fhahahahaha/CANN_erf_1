#!/bin/bash
# Erf算子工程编译脚本
set -e

if [ -z "$ASCEND_HOME_PATH" ]; then
    if [ -n "$ASCEND_TOOLKIT_HOME" ]; then
        export ASCEND_HOME_PATH="$ASCEND_TOOLKIT_HOME"
    else
        echo "please set ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH env."
        exit 1
    fi
fi
echo "using ASCEND_HOME_PATH: $ASCEND_HOME_PATH"

SCRIPT_DIR="$(realpath "$(dirname "$0")")"
BUILD_DIR="$SCRIPT_DIR/build_out"

# 清理旧的构建目录
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# 使用 CMakePresets.json 中的 default preset 配置
cmake -S . -B "$BUILD_DIR" --preset=default

# 先编译 kernel 二进制
cmake --build "$BUILD_DIR" --target binary -j$(nproc)

# 再打包生成 .run 安装包
cmake --build "$BUILD_DIR" --target package -j$(nproc)

echo ""
echo "Build completed."
echo "算子安装包:"
ls -la "$BUILD_DIR"/*.run 2>/dev/null || echo "未找到 .run 文件, 请检查编译日志"
