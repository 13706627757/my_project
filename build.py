import tensorrt as trt
import pycuda.driver as cuda
import common
import pycuda.autoinit
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    # 创建builder和network
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(common.EXPLICIT_BATCH)
    config = builder.create_builder_config()

    # 设置工作空间大小
    config.max_workspace_size = 1 << 30  # 1GB的工作空间

    # 如果你需要设置FP16或INT8的优化，请在这里配置
    # if builder.platform_has_fast_fp16:
    #     config.set_flag(trt.BuilderFlag.FP16)

    # 解析器用来解析.onnx文件
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 加载ONNX文件并解析
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    # 使用解析器的输出构建engine
    engine = builder.build_engine(network, config)

    # 保存engine到文件
    with open(engine_file_path, 'wb') as f:
        f.write(engine.serialize())

    return engine
onnx_path = "model.onnx"
engine_path = "model.engine"
build_engine(onnx_path, engine_path)
