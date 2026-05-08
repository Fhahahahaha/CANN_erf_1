/**
 * @file main.cpp
 * @brief Erf算子单算子API测试 (透传模式: output = input)
 */
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "aclnn_erf.h"

#define SUCCESS 0
#define FAILED 1

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

int64_t GetShapeSize(const std::vector<int64_t> &shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

template <typename T>
int CreateAclTensor(const std::vector<T> &hostData, const std::vector<int64_t> &shape, void **deviceAddr,
                    aclDataType dataType, aclTensor **tensor)
{
    auto size = GetShapeSize(shape) * sizeof(T);
    auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return FAILED);
    ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return FAILED);
    *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, nullptr, 0, aclFormat::ACL_FORMAT_ND,
                              shape.data(), shape.size(), *deviceAddr);
    return SUCCESS;
}

void DestroyAclTensor(aclTensor *tensor, void *deviceAddr)
{
    if (tensor != nullptr) aclDestroyTensor(tensor);
    if (deviceAddr != nullptr) aclrtFree(deviceAddr);
}

int RunOneCase(const std::vector<int64_t> &shape,
               const std::vector<float> &inputData,
               const std::vector<float> &goldenData,
               const char *caseName,
               aclrtStream stream)
{
    size_t n = GetShapeSize(shape);
    std::vector<float> outputHostData(n, -1.0f);

    void *inDev = nullptr, *outDev = nullptr;
    aclTensor *inTensor = nullptr, *outTensor = nullptr;
    auto ret = CreateAclTensor(inputData, shape, &inDev, aclDataType::ACL_FLOAT, &inTensor);
    CHECK_RET(ret == ACL_SUCCESS, return FAILED);
    ret = CreateAclTensor(outputHostData, shape, &outDev, aclDataType::ACL_FLOAT, &outTensor);
    CHECK_RET(ret == ACL_SUCCESS, DestroyAclTensor(inTensor, inDev); return FAILED);

    uint64_t wsSize = 0;
    aclOpExecutor *executor = nullptr;
    ret = aclnnErfGetWorkspaceSize(inTensor, outTensor, &wsSize, &executor);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] GetWorkspaceSize failed ret=%d\n", caseName, ret); return FAILED);

    void *wsAddr = nullptr;
    if (wsSize > 0) {
        ret = aclrtMalloc(&wsAddr, wsSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] ws malloc failed\n", caseName); return FAILED);
    }

    ret = aclnnErf(wsAddr, wsSize, executor, stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] aclnnErf failed ret=%d\n", caseName, ret); return FAILED);

    ret = aclrtSynchronizeStream(stream);
    CHECK_RET(ret == ACL_SUCCESS,
              LOG_PRINT("[%s] sync failed ret=%d\n", caseName, ret); return FAILED);

    ret = aclrtMemcpy(outputHostData.data(), n * sizeof(float), outDev, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("[%s] memcpy failed ret=%d\n", caseName, ret); return FAILED);

    if (wsAddr != nullptr) aclrtFree(wsAddr);
    DestroyAclTensor(inTensor, inDev);
    DestroyAclTensor(outTensor, outDev);

    // 统计mismatch并打印前10个
    int64_t mismatchCount = 0;
    int64_t firstGood = -1;
    for (size_t i = 0; i < n; i++) {
        if (outputHostData[i] != goldenData[i]) {
            if (mismatchCount < 5) {
                LOG_PRINT("  [MIS @%zu] got=%.6f expect=%.6f\n", i, outputHostData[i], goldenData[i]);
            }
            mismatchCount++;
        } else if (firstGood < 0) {
            firstGood = static_cast<int64_t>(i);
        }
    }

    LOG_PRINT("[%s] shape=[%lld,%lld] total=%zu\n", caseName,
              static_cast<long long>(shape[0]), static_cast<long long>(shape[1]), n);
    LOG_PRINT("  first 8 output: ");
    for (size_t i = 0; i < std::min<size_t>(8, n); i++) LOG_PRINT("%.2f ", outputHostData[i]);
    LOG_PRINT("\n  last  8 output: ");
    for (size_t i = (n > 8 ? n - 8 : 0); i < n; i++) LOG_PRINT("%.2f ", outputHostData[i]);
    LOG_PRINT("\n  first good @%lld, mismatches=%lld/%zu\n",
              static_cast<long long>(firstGood), static_cast<long long>(mismatchCount), n);

    if (mismatchCount == 0) {
        LOG_PRINT("[%s] PASS\n\n", caseName);
        return SUCCESS;
    } else {
        LOG_PRINT("[%s] FAIL\n\n", caseName);
        return FAILED;
    }
}

int main(int argc, char **argv)
{
    auto ret = aclInit(nullptr);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. %d\n", ret); return FAILED);
    ret = aclrtSetDevice(0);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("setDevice failed. %d\n", ret); return FAILED);

    aclrtStream stream;
    ret = aclrtCreateStream(&stream);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("createStream failed. %d\n", ret); return FAILED);

    int32_t overallResult = SUCCESS;

    // Case 1: [8, 2048] = 16384 elements, input=1.0, expect 1.0 everywhere
    {
        std::vector<int64_t> shape = {8, 2048};
        size_t n = GetShapeSize(shape);
        std::vector<float> inputData(n, 1.0f);
        std::vector<float> goldenData(n, 1.0f);
        if (RunOneCase(shape, inputData, goldenData, "Case1", stream) != SUCCESS) {
            overallResult = FAILED;
        }
    }

    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();

    LOG_PRINT("========================================\n");
    LOG_PRINT("%s\n", overallResult == SUCCESS ? "PASS" : "FAIL");
    LOG_PRINT("========================================\n");
    return overallResult;
}
