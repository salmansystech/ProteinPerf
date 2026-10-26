"""Hardware and software details recorded with every result"""
import platform
import torch

def environment_info() -> dict:
    """GPU, driver and library versions, so results can be compared and reproduced"""
    import transformers

    info = {
        "python": platform.python_version(),
        "os": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update({
            "gpu": props.name,
            "gpu_memory_gb": round(props.total_memory / 2**30, 1),
            "compute_capability": f"{props.major}.{props.minor}",
            "multiprocessors": props.multi_processor_count,
            "cuda": torch.version.cuda,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        })
    return info
