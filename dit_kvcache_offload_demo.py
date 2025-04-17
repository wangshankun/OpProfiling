import torch
import time

import inspect
import os.path
from copy import deepcopy
def printf(*args, **kwargs):
    """
    An enhanced print function that includes the file name and line number.

    Args:
        *args: The values to be printed.
        **kwargs: The keyword arguments to be passed to the built-in print function.
    """
    # Get the current file name and line number
    frame = inspect.currentframe().f_back
    file_name = os.path.basename(frame.f_code.co_filename)
    line_number = frame.f_lineno

    # Construct the output string
    output = f"{file_name}:{line_number} - "
    output += " ".join(str(arg) for arg in args)

    # Print the output
    print(output, **kwargs)

# 参数配置
num_layers = 40      # Transformer 层数
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备选择

g2c_streams = torch.cuda.Stream(priority=-2)  # 优先级最高
c2g_streams = torch.cuda.Stream(priority=-1)  # 次高
compute_streams = torch.cuda.Stream(priority=0)  # 默认优先级，最低
# 模拟的前向计算函数（在真实场景中替换为模型调用）
def model_forward(x, y):
    # 这里用 sleep 模拟耗时
    time.sleep(0.1)
    y['k'] = x + y['k']
    y['v'] = x + y['v']
    return y


seq_len = 75600
kvcache = []

torch.manual_seed(42)
test_gpu_input = torch.randn(
    [1, 40, seq_len, 128],
    dtype=torch.float16, device=device)

printf(f"test_gpu_input[0]:{test_gpu_input[0][0][0][0]}, test_gpu_input device:{test_gpu_input.device} ")
for _ in range(num_layers):
    kvcache.append({#偶数kvcache在GPU在里面，奇数kvcache在CPU里面
        "k": torch.zeros(
            [1, 40, seq_len, 128],
            dtype=torch.float16, device=device),
        "v": torch.zeros(
            [1, 40, seq_len, 128],
            dtype=torch.float16, device=device),
    })
    kvcache.append({
        "k": torch.zeros(
            [1, 40, seq_len, 128],
            dtype=torch.float16, device='cpu', pin_memory=True
        ),
        "v": torch.zeros(
            [1, 40, seq_len, 128],
            dtype=torch.float16, device='cpu', pin_memory=True
        ),
    })

compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

# 计时开始
start = time.time()
for _ in range(20):
    sss = time.time()
    for i in range(num_layers):
        evt_transfer_done = torch.cuda.Event()
        evt_compute_done  = torch.cuda.Event()
        if i % 2 == 0:
            ss = time.time()
            ###using time 0.17s
            with torch.cuda.stream(c2g_streams):#预取奇数需要的数据给GPU
                kvcache[i+1]['k'] = kvcache[i+1]['k'].to(device=device, non_blocking=True)
                kvcache[i+1]['v'] = kvcache[i+1]['v'].to(device=device, non_blocking=True)
                evt_transfer_done.record(c2g_streams)
            ###using time 0.1s
            with torch.cuda.stream(compute_streams):
                kvcache[i] = model_forward(test_gpu_input, kvcache[i])
            printf(f"i:{i}, using time:{time.time() - ss}")
        else:
            ss = time.time()
            with torch.cuda.stream(transfer_stream):
                if i > 1:
                    transfer_stream.wait_event(evt_compute_done)
                    kvcache[i-2]['k'] = kvcache[i-2]['k'].to(device='cpu', non_blocking=True)#存储上一个计算结果从gpu到cpu中
                    kvcache[i-2]['v'] = kvcache[i-2]['v'].to(device='cpu', non_blocking=True)#存储上一个计算结果从gpu到cpu中

            with torch.cuda.stream(compute_streams):
                compute_stream.wait_event(evt_transfer_done)
                kvcache[i] = model_forward(test_gpu_input, kvcache[i])#计算本次结果存储在gpu上
                evt_compute_done.record(compute_stream)
            printf(f"i:{i}, using time:{time.time() - ss}")
        
            if i % 21 == 0:
                printf(f"kvcache[i-2]['k']:{kvcache[i-2]['k'].device} value:{kvcache[i-2]['k'][0][0][0][0]}")
                printf(f"kvcache[i]['k']:{kvcache[i]['k'].device} value:{kvcache[i]['k'][0][0][0][0]}")
    torch.cuda.synchronize()
    print(f"Layer {i} processed in {time.time() - sss:.3f}s")
    # 等待所有 stream 完成
print(f"Total elapsed time: {time.time() - start:.3f}s")
