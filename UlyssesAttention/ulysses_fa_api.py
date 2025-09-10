import torch
import torch.distributed as dist

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    logger.info("flash_attn_varlen_func_v3 not found, please install flash_attn3 first")
    flash_attn_varlen_func_v3 = None


import torch.profiler as profiler

def profile_flash_attn(local_device="cuda", nc=5, seqlen=5400, dim=128, n_iter=50):
    # 构造输入
    q_attn = torch.randn(1, nc, seqlen, dim, device=local_device, dtype=torch.bfloat16)
    k_attn = torch.randn(1, nc, seqlen, dim, device=local_device, dtype=torch.bfloat16)
    v_attn = torch.randn(1, nc, seqlen, dim, device=local_device, dtype=torch.bfloat16)

    # ---------- Warmup ----------
    for _ in range(5):
        _ = flash_attn_varlen_func_v3(q_attn, k_attn, v_attn, None, None, None, None)

    torch.cuda.synchronize()

    # ---------- 用 cuda.Event 统计 ----------
    times = []
    for _ in range(n_iter):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        attn = flash_attn_varlen_func_v3(
            q_attn, k_attn, v_attn,
            None, None, None, None
        )
        end_event.record()

        torch.cuda.synchronize()
        elapsed = start_event.elapsed_time(end_event)  # ms
        times.append(elapsed)

    avg_event = sum(times) / len(times)
    print(f"[cuda.Event] Average: {avg_event:.3f} ms, shape={attn.shape}")

    # ---------- 用 torch.profiler 统计 ----------
    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False
    ) as prof:
        for _ in range(n_iter):
            attn_prof = flash_attn_varlen_func_v3(
                q_attn, k_attn, v_attn,
                None, None, None, None
            ).transpose(1, 2).squeeze(0).reshape(seqlen, -1)

    print(prof.key_averages(group_by_stack_n=10).table(
        sort_by="cuda_time_total", row_limit=4
    ))

    return attn, avg_event, prof


profile_flash_attn(nc=5)#8cards GPU
profile_flash_attn(nc=10)
profile_flash_attn(nc=20)
profile_flash_attn(nc=40)#single GPU
