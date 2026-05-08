// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include "graph/utils/type_utils.h"
#include "../op_kernel/erf_tiling.h"
// #include "../op_kernel/tiling_key_erf.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // 获取平台信息
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        uint32_t coreNum = static_cast<uint32_t>(platform.GetCoreNumAiv());

        // 获取算子输入信息
        uint32_t inputNum = context->GetInputShape(0)->GetStorageShape().GetShapeSize();
        uint32_t typeLength = 0;
        ge::TypeUtils::GetDataTypeLength(context->GetInputDesc(0)->GetDataType(), typeLength);
        uint32_t inputLength = inputNum * typeLength;
    
        // // 配置tiling key, 实现kernel侧不同数据类型的区分
        // uint32_t DT_X = static_cast<uint32_t>(dtype_x);
        // ASCENDC_TPL_SEL_PARAM(context, DT_X);

        /* 首先对齐到32B */
        const uint32_t BLOCK_SIZE = 32;
        uint32_t inputLengthAlgin32 = (((inputLength + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE);
        /* 计算要使用的核心数 */
        coreNum = std::min(coreNum, inputLengthAlgin32 / BLOCK_SIZE);
        coreNum = std::max(coreNum, static_cast<uint32_t>(1));

        uint32_t everyCoreInputBlockNum = inputLengthAlgin32 / BLOCK_SIZE / coreNum;
        uint32_t tailBlockNum = (inputLengthAlgin32 / BLOCK_SIZE) % coreNum;
        context->SetBlockDim(coreNum);

        /* 获取UB大小，从而确定对算子内数据切分 */
        uint64_t ubSize;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        uint32_t ubDataNumber = 3;
        uint32_t tileBlockNum = (ubSize / BLOCK_SIZE ) / ubDataNumber;
        uint32_t tileDataNum = (tileBlockNum * BLOCK_SIZE) / typeLength;

        /* 分配大小核处理的数据量（因为对齐问题，需要处理多一个数据的核就是大核） */
        uint32_t smallCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
        uint32_t smallTileNum = everyCoreInputBlockNum / tileBlockNum;
        uint32_t finalSmallTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? smallTileNum : smallTileNum + 1;
        uint32_t smallTailDataNum = smallCoreDataNum - (tileDataNum * smallTileNum);
        smallTailDataNum = smallTailDataNum == 0 ? tileDataNum : smallTailDataNum;
        // 大核
        everyCoreInputBlockNum += 1;
        uint32_t bigCoreDataNum = everyCoreInputBlockNum * BLOCK_SIZE / typeLength;
        uint32_t bigTileNum = everyCoreInputBlockNum / tileBlockNum;
        uint32_t finalBigTileNum = (everyCoreInputBlockNum % tileBlockNum) == 0 ? bigTileNum : bigTileNum + 1;
        uint32_t bigTailDataNum = bigCoreDataNum - tileDataNum * bigTileNum;
        bigTailDataNum = bigTailDataNum == 0 ? tileDataNum : bigTailDataNum;

        /* 填充结构体 */
        ErfTilingData *tiling = context->GetTilingData<ErfTilingData>();
        tiling->smallCoreDataNum = smallCoreDataNum;
        tiling->bigCoreDataNum = bigCoreDataNum;
        tiling->tileDataNum = tileDataNum;
        tiling->smallTailDataNum = smallTailDataNum;
        tiling->bigTailDataNum = bigTailDataNum;
        tiling->finalSmallTileNum = finalSmallTileNum;
        tiling->finalBigTileNum = finalBigTileNum;
        tiling->tailBlockNum = tailBlockNum;

        // 配置workspace大小 (本算子无需额外workspace)
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        currentWorkspace[0] = 0;
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *inputShape = context->GetInputShape(0);
        gert::Shape *outputShape = context->GetOutputShape(0);
        *outputShape = *inputShape;
        return GRAPH_SUCCESS;
    }
    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        context->SetOutputDataType(0, context->GetInputDataType(0));
        return ge::GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class Erf : public OpDef {
    public:
        explicit Erf(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->Output("y")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND});
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(Erf);
}  // namespace ops
