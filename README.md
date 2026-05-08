# CANN Erf 算子工程

基于 Ascend C 开发的 Erf 算子工程，支持在昇腾 910B 上运行。

## 目录结构

```
├── build.sh                  # 算子工程编译脚本
├── CMakeLists.txt             # 根 CMake 配置
├── CMakePresets.json          # CMake 预设
├── framework/                 # 框架插件目录
├── op_host/                   # Host 侧实现
│   ├── CMakeLists.txt
│   └── erf.cpp                # Tiling、Shape/Dtype推导、算子注册
├── op_kernel/                 # Kernel 侧实现
│   ├── CMakeLists.txt
│   ├── erf.cpp                # 核函数实现
│   └── erf_tiling.h           # Tiling 结构体定义
└── test/                      # 测试
    ├── main.cpp               # 单算子 API 调用测试
    └── run_test.sh            # 一键编译部署测试脚本
```

## 构建与测试

```bash
# 编译算子工程
bash build.sh

# 一键编译部署并测试
cd test && bash run_test.sh
```

## 算子规格

- 输入: x (DT_FLOAT, FORMAT_ND)
- 输出: y (DT_FLOAT, FORMAT_ND)
- 支持设备: ascend910b
