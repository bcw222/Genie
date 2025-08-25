import subprocess
import re


def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version'], stderr=subprocess.STDOUT).decode()
        match = re.search(r'V(\d+\.\d+\.\d+)', output)
        if match:
            return match.group(1)  # 例如 12.8.61
    except Exception as e:
        return f"Error: {e}"
