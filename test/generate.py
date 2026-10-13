import numpy
from safetensors.numpy import save_file

tensors: dict[str, numpy.ndarray] = {
    "f16_tensor": numpy.array(range(24), dtype=numpy.float16).reshape(2, 3, 4),
    "f32_tensor": numpy.array(range(24), dtype=numpy.float32).reshape(2, 3, 4),
    "f64_tensor": numpy.array(range(24), dtype=numpy.float64).reshape(2, 3, 4),
    "i8_tensor": numpy.array(range(24), dtype=numpy.int8).reshape(2, 3, 4),
    "u8_tensor": numpy.array(range(24), dtype=numpy.uint8).reshape(2, 3, 4),
    "i16_tensor": numpy.array(range(24), dtype=numpy.int16).reshape(2, 3, 4),
    "u16_tensor": numpy.array(range(24), dtype=numpy.uint16).reshape(2, 3, 4),
    "i32_tensor": numpy.array(range(24), dtype=numpy.int32).reshape(2, 3, 4),
    "u32_tensor": numpy.array(range(24), dtype=numpy.uint32).reshape(2, 3, 4),
    "i64_tensor": numpy.array(range(24), dtype=numpy.int64).reshape(2, 3, 4),
    "u64_tensor": numpy.array(range(24), dtype=numpy.uint64).reshape(2, 3, 4),
    "bool_tensor": numpy.array([], dtype=numpy.bool),
}
metadata = {"metadata_key": "metadata_value"}
save_file(tensors, "tensors.safetensors", metadata)
