import numpy as np


def convert(
        fp16_bin_path: str, output_fp32_bin_path: str
):
    fp16_array = np.fromfile(fp16_bin_path, dtype=np.float16)
    fp32_array = fp16_array.astype(np.float32)
    fp32_array.tofile(output_fp32_bin_path)
    pass


def restore_fp32_weights_from_fp16(model_dir: str):
    pass
