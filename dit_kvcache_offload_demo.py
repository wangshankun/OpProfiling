import torch
import time

import inspect
import os.path
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
seq_len = 32760

dtype      = torch.float16
block_num = 7 #causal block attention num
bfs_len   = seq_len // block_num#blockfreams_seq_length

torch.manual_seed(42)
test_gpu_input = torch.randn(
    [1, 40, seq_len, 128],
    dtype=dtype, device=device)

g2c_streams = torch.cuda.Stream(priority=-2)  # 优先级最高
c2g_streams = torch.cuda.Stream(priority=-1)  # 次高
compute_streams = torch.cuda.Stream(priority=0)  # 默认优先级，最低

# 模拟的前向计算函数（在真实场景中替换为模型调用）
def model_forward(x, y, block_id):
    # 这里用 sleep 模拟耗时wan 480p模型执行一层TransformBlock的时间
    time.sleep(0.022)

    #将y的前bidx个块，按照dim=2的维度合并
    end_block = block_id + 1
    merged_y_k = torch.cat(
        [y[i]['k'] for i in range(end_block)],
        dim=2  # 沿时间维拼接
    )

    merged_y_v = torch.cat(
        [y[i]['v'] for i in range(end_block)],
        dim=2  # 沿时间维拼接
    )

    bfs_len = y[0]['k'].shape[2]#每个block里面帧tokens的长度
    #做加法
    merged_y_k = x[:,:,:end_block*bfs_len,:] + merged_y_k
    merged_y_v = x[:,:,:end_block*bfs_len,:] + merged_y_v

    # 拆分
    split_y_k = torch.split(merged_y_k, bfs_len, dim=2)  # 返回一个 tuple，每块大小为bfs_len
    split_y_v = torch.split(merged_y_v, bfs_len, dim=2) 
    # Step 4: 放回字典中
    for i in range(end_block):
        y[i]['k'] = split_y_k[i]
        y[i]['v'] = split_y_v[i]

    return y

kvcache = []

for layer_idx in range(num_layers):
    cache_block = {}
    for block_idx in range(block_num):
        cache_block[block_idx] = {#偶数kvcache在GPU在里面，奇数kvcache在CPU里面
            "k": torch.zeros(
                [1, 40, bfs_len, 128],
                dtype=dtype, device=device if layer_idx % 2 == 0 else "cpu"
            ),
            "v": torch.zeros(
                [1, 40, bfs_len, 128],
                dtype=dtype, device=device if layer_idx % 2 == 0 else "cpu"
            ),
        }
    kvcache.append(cache_block)

print(f"kvcache[0][0]['k'].device: {kvcache[0][0]['k'].device}")
print(f"kvcache[1][0]['k'].device: {kvcache[1][0]['k'].device}")
compute_stream = torch.cuda.Stream()
transfer_stream = torch.cuda.Stream()

# 计时开始
start = time.time()
for block_id in range(block_num):
    sss = time.time()
    for _ in range(20):
        for i in range(num_layers):
            evt_transfer_done = torch.cuda.Event()
            evt_compute_done  = torch.cuda.Event()
            if i % 2 == 0:
                with torch.cuda.stream(c2g_streams):#预取奇数需要的数据给GPU
                    for bidx in range(block_id+1):
                        kvcache[i+1][bidx]['k'] = kvcache[i+1][bidx]['k'].to(device=device, non_blocking=True)
                        kvcache[i+1][bidx]['v'] = kvcache[i+1][bidx]['v'].to(device=device, non_blocking=True)
                    evt_transfer_done.record(c2g_streams)
                with torch.cuda.stream(compute_streams):
                    kvcache[i] = model_forward(test_gpu_input, kvcache[i], block_id)
            else:
                with torch.cuda.stream(transfer_stream):
                    if i > 1:
                        transfer_stream.wait_event(evt_compute_done)
                        for bidx in range(block_id+1):
                            kvcache[i-2][bidx]['k'] = kvcache[i-2][bidx]['k'].to(device='cpu', non_blocking=True)
                            kvcache[i-2][bidx]['v'] = kvcache[i-2][bidx]['v'].to(device='cpu', non_blocking=True)
                with torch.cuda.stream(compute_streams):
                    compute_stream.wait_event(evt_transfer_done)
                    kvcache[i] = model_forward(test_gpu_input, kvcache[i], block_id)#计算本次结果存储在gpu上
                    evt_compute_done.record(compute_stream)
        torch.cuda.synchronize()
    print(f"block_id {block_id} processed in {time.time() - sss:.3f}s")
    # 等待所有 stream 完成
print(f"Total elapsed time: {time.time() - start:.3f}s")
