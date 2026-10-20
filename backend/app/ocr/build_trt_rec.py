import os
import sys
import time
from pathlib import Path
import tensorrt as trt

def build_engine(
    onnx_path: str,
    engine_path: str,
    fp16: bool = True,
    workspace_gb: int = 1,
):
    print(f"Building TensorRT engine from {onnx_path} -> {engine_path}...")
    t0 = time.time()
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    
    # Workspace memory limit (1GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb * (1 << 30))
    
    if fp16 and builder.platform_has_fast_fp16:
        print("Enabling FP16 precision mode")
        config.set_flag(trt.BuilderFlag.FP16)
        
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Failed to parse ONNX:")
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            sys.exit(1)
            
    # Set dynamic optimization profile for input tensor 'x'
    input_tensor = network.get_input(0)
    print(f"Configuring dynamic profile for input '{input_tensor.name}'...")
    profile = builder.create_optimization_profile()
    profile.set_shape(
        input_tensor.name,
        min=(1, 3, 48, 64),
        opt=(1, 3, 48, 320),
        max=(4, 3, 48, 960)
    )
    config.add_optimization_profile(profile)
    
    print("Serializing TensorRT engine (this may take 2-3 minutes)...")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("Failed to build serialized engine!")
        sys.exit(1)
        
    Path(engine_path).parent.mkdir(parents=True, exist_ok=True)
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
        
    print(f"TensorRT engine successfully built in {round(time.time() - t0, 2)}s: {engine_path}")

if __name__ == '__main__':
    onnx_file = '/home/arctec/Downloads/sarweshar/backend/.venv/lib/python3.8/site-packages/rapidocr_onnxruntime/models/ch_PP-OCRv4_rec_infer.onnx'
    engine_file = '/home/arctec/Downloads/sarweshar/backend/models/rec_ppocrv4.engine'
    build_engine(onnx_file, engine_file)
