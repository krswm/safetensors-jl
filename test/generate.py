import numpy
from safetensors.numpy import save_file

tensors: dict[str, numpy.ndarray] = {
    "F32_tensor": numpy.array(range(24), dtype=numpy.float32).reshape(2, 3, 4),
}
save_file(tensors, "tensors.safetensors")
