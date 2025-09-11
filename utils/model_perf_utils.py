import time

import pynvml
import torch
from tqdm import trange


def get_gpu_memory_usage(device_idx: int | None = None):
    pynvml.nvmlInit()
    if device_idx is None:
        device_idx = torch.cuda.current_device()
    handle = pynvml.nvmlDeviceGetHandleByIndex(device_idx)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"Total GPU memory: {info.total / 1024**3:.2f} GB")
    print(f"Used GPU memory: {info.used / 1024**3:.2f} GB")
    print(f"Free GPU memory: {info.free / 1024**3:.2f} GB")
    pynvml.nvmlShutdown()


def measure_throughput(
    model: "torch.nn.Module",
    input_size: "list[tuple[int, ...]] | tuple[int, ...] | None",
    batch_size: int,
    num_warmup: int = 10,
    num_iterations: int = 50,
):
    device = next(model.parameters()).device

    device_idx = int(str(device)[-1])
    get_gpu_memory_usage(device_idx)

    model.eval()

    if isinstance(input_size[0], tuple):
        dummy_input = []
        for inp_shape in input_size:
            if inp_shape is None:
                inp = torch.randn(batch_size, device=device)
            else:
                inp = torch.randn(batch_size, *inp_shape, device=device)
            dummy_input.append(inp)
    elif isinstance(input_size, (list, tuple)):
        dummy_input = [torch.randn(batch_size, *input_size, device=device)]
    elif input_size is None:
        dummy_input = [torch.randn(batch_size, device=device)]
    else:
        raise ValueError(
            f"Invalid input size: {input_size}, should be list[tuple[int, ...]] | tuple[int, ...] | None"
        )

    print(f"Warming up {num_warmup} iterations ...")
    with torch.no_grad():
        for _ in trange(num_warmup):
            model(*dummy_input)

    torch.cuda.synchronize()

    print(f"Measuring model for {num_iterations} iterations ...")
    total_time = 0
    with torch.no_grad():
        for _ in trange(num_iterations):
            start_time = time.time()
            model(*dummy_input)
            end_time = time.time()
            total_time += end_time - start_time

    images_per_second = (num_iterations * batch_size) / total_time
    print(f"Throughput: {images_per_second:.2f} images/second")
    return images_per_second
