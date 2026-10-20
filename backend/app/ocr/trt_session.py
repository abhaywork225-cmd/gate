import os
import threading
from pathlib import Path
from typing import Any, List, Optional, Union
import numpy as np
import tensorrt as trt
import torch

class TensorRTInferSession:
    """High performance TensorRT GPU Execution Session for RapidOCR models with CPU fallback."""
    def __init__(self, engine_path: Union[str, Path], fallback_session: Optional[Any] = None):
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found at {self.engine_path}")
            
        self.fallback_session = fallback_session
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
            
        self.context = self.engine.create_execution_context()
        self._lock = threading.Lock()
        
        # Query profile 0 min and max shapes for input 0
        try:
            profile = self.engine.get_profile_shape(0, 0)
            self.min_shape = profile[0]
            self.opt_shape = profile[1]
            self.max_shape = profile[2]
        except Exception:
            self.min_shape = (1, 3, 48, 64)
            self.max_shape = (4, 3, 48, 960)
            
    def _run_single_chunk(self, chunk: np.ndarray) -> np.ndarray:
        shape = chunk.shape
        self.context.set_binding_shape(0, shape)
        
        input_tensor = torch.from_numpy(chunk).cuda()
        out_shape = tuple(self.context.get_binding_shape(1))
        output_tensor = torch.empty(out_shape, dtype=torch.float32, device='cuda')
        
        self.context.execute_v2([int(input_tensor.data_ptr()), int(output_tensor.data_ptr())])
        torch.cuda.synchronize()
        return output_tensor.cpu().numpy()
        
    def __call__(self, input_data: np.ndarray) -> List[np.ndarray]:
        if not input_data.flags.c_contiguous or input_data.dtype != np.float32:
            input_data = np.ascontiguousarray(input_data, dtype=np.float32)
            
        shape = input_data.shape
        # Check if height and width fit within the TRT engine profile
        if len(shape) == 4:
            b, c, h, w = shape
            min_b, min_c, min_h, min_w = self.min_shape
            max_b, max_c, max_h, max_w = self.max_shape
            
            # If dimensions out of bounds and fallback is available, run fallback
            if (h < min_h or h > max_h or w < min_w or w > max_w):
                if self.fallback_session is not None:
                    return self.fallback_session(input_data)
                    
            with self._lock:
                try:
                    # If batch size exceeds engine's max batch size, process in chunks
                    if b > max_b:
                        results = []
                        for i in range(0, b, max_b):
                            sub_chunk = input_data[i:i + max_b]
                            results.append(self._run_single_chunk(sub_chunk))
                        return [np.concatenate(results, axis=0)]
                    else:
                        return [self._run_single_chunk(input_data)]
                except Exception as e:
                    if self.fallback_session is not None:
                        return self.fallback_session(input_data)
                    raise e
                    
        with self._lock:
            try:
                return [self._run_single_chunk(input_data)]
            except Exception as e:
                if self.fallback_session is not None:
                    return self.fallback_session(input_data)
                raise e
