import torch
import torch.distributed as dist

try:
    from flash_attn_interface import flash_attn_varlen_func as flash_attn_varlen_func_v3
except ImportError:
    logger.info("flash_attn_varlen_func_v3 not found, please install flash_attn3 first")
    flash_attn_varlen_func_v3 = None

from torch.profiler import record_function
from torch.profiler import profile, record_function, ProfilerActivity

from loguru import logger


def all2all_seq2head(input, group=None):
    """
    将输入张量从 [seq_len/N, heads, hidden_dims] 转换为 [seq_len, heads/N, hidden_dims] 的格式。

    参数:
        input (torch.Tensor): 输入张量，形状为 [seq_len/N, heads, hidden_dims]

    返回:
        torch.Tensor: 转换后的输出张量，形状为 [seq_len, heads/N, hidden_dims]
    """
    # 确保输入是一个3D张量
    assert input.dim() == 3, f"input must be 3D tensor"

    # 获取当前进程的世界大小
    world_size = dist.get_world_size(group=group)

    # 获取输入张量的形状
    shard_seq_len, heads, hidden_dims = input.shape
    seq_len = shard_seq_len * world_size  # 计算总序列长度
    shard_heads = heads // world_size  # 计算每个进程处理的头数

    # 重塑输入张量以便进行 all-to-all 操作
    input_t = (
        input.reshape(shard_seq_len, world_size, shard_heads, hidden_dims)  # 重塑为 [shard_seq_len, world_size, shard_heads, hidden_dims]
        .transpose(0, 1)  # 转置以便进行 all-to-all 操作
        .contiguous()  # 确保内存连续
    )

    # 创建一个与输入张量相同形状的输出张量
    output = torch.empty_like(input_t)

    # 执行 all-to-all 操作，将输入张量的内容分发到所有进程
    dist.all_to_all_single(output, input_t, group=group)

    # 重塑输出张量为 [seq_len, heads/N, hidden_dims] 形状
    output = output.reshape(seq_len, shard_heads, hidden_dims).contiguous()

    return output  # 返回转换后的输出张量

 
def all2all_head2seq(input, group=None):
    """
    将输入张量从 [seq_len, heads/N, hidden_dims] 转换为 [seq_len/N, heads, hidden_dims] 的格式。

    参数:
        input (torch.Tensor): 输入张量，形状为 [seq_len, heads/N, hidden_dims]

    返回:
        torch.Tensor: 转换后的输出张量，形状为 [seq_len/N, heads, hidden_dims]
    """
    # 确保输入是一个3D张量
    assert input.dim() == 3, f"input must be 3D tensor"

    # 获取当前进程的世界大小
    world_size = dist.get_world_size(group=group)

    # 获取输入张量的形状
    seq_len, shard_heads, hidden_dims = input.shape
    heads = shard_heads * world_size  # 计算总头数
    shard_seq_len = seq_len // world_size  # 计算每个进程处理的序列长度

    # 重塑输入张量以便进行 all-to-all 操作
    input_t = (
        input.reshape(world_size, shard_seq_len, shard_heads, hidden_dims)  # 重塑为 [world_size, shard_seq_len, shard_heads, hidden_dims]
        .transpose(1, 2)  # 转置以便进行 all-to-all 操作
        .contiguous()  # 确保内存连续
        .reshape(world_size, shard_heads, shard_seq_len, hidden_dims)  # 再次重塑为 [world_size, shard_heads, shard_seq_len, hidden_dims]
    )

    # 创建一个与输入张量相同形状的输出张量
    output = torch.empty_like(input_t)

    # 执行 all-to-all 操作，将输入张量的内容分发到所有进程
    dist.all_to_all_single(output, input_t, group=group)

    # 重塑输出张量为 [heads, shard_seq_len, hidden_dims] 形状
    output = output.reshape(heads, shard_seq_len, hidden_dims)

    # 转置输出张量并重塑为 [shard_seq_len, heads, hidden_dims] 形状
    output = output.transpose(0, 1).contiguous().reshape(shard_seq_len, heads, hidden_dims)

    return output  # 返回转换后的输出张量


class UlyssesAttnWeight():
 
    def __init__(self):
        self.count = 0

    def apply(
        self, q, k, v, img_qkv_len, cu_seqlens_qkv,
        attention_module=None, seq_p_group=None, model_cls=None
    ):
        with record_function("ulysses/init"):
            world_size = dist.get_world_size(seq_p_group)
            cur_rank = dist.get_rank(seq_p_group)
            seq_len = q.shape[0]

            if len(cu_seqlens_qkv) == 3:
                txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len
                txt_mask_len = cu_seqlens_qkv[2] - img_qkv_len
            elif len(cu_seqlens_qkv) == 2:
                txt_qkv_len = cu_seqlens_qkv[1] - img_qkv_len
                txt_mask_len = None

            _, heads, hidden_dims = q.shape
            shard_heads = heads // world_size
            shard_seqlen = img_qkv_len

        with record_function("ulysses/split_qkv"):
            img_q, img_k, img_v = (
                q[:img_qkv_len].contiguous(),
                k[:img_qkv_len].contiguous(),
                v[:img_qkv_len].contiguous(),
            )
            txt_q, txt_k, txt_v = (
                q[img_qkv_len:].contiguous(),
                k[img_qkv_len:].contiguous(),
                v[img_qkv_len:].contiguous(),
            )

        with record_function("ulysses/all2all_seq2head"):
            img_q = all2all_seq2head(img_q, group=seq_p_group)
            img_k = all2all_seq2head(img_k, group=seq_p_group)
            img_v = all2all_seq2head(img_v, group=seq_p_group)
            torch.cuda.synchronize()

        with record_function("ulysses/select_txt_heads"):
            txt_q = txt_q[:, cur_rank * shard_heads:(cur_rank + 1) * shard_heads, :]
            txt_k = txt_k[:, cur_rank * shard_heads:(cur_rank + 1) * shard_heads, :]
            txt_v = txt_v[:, cur_rank * shard_heads:(cur_rank + 1) * shard_heads, :]

        with record_function("ulysses/concat_qkv"):
            q = torch.cat((img_q, txt_q), dim=0)
            k = torch.cat((img_k, txt_k), dim=0)
            v = torch.cat((img_v, txt_v), dim=0)

        with record_function("ulysses/build_cu_seqlens"):
            cu_seqlens_qkv = torch.zeros([2], dtype=torch.int32, device="cuda")
            s = txt_qkv_len + img_q.shape[0]
            cu_seqlens_qkv[1] = s
            if txt_mask_len:
                s2 = txt_mask_len + img_q.shape[0]
                cu_seqlens_qkv = torch.cat([cu_seqlens_qkv, torch.tensor([s2], device="cuda")])
            max_seqlen_qkv = img_q.shape[0] + txt_q.shape[0]

        with record_function("ulysses/attention"):
            # Reshape for scaled_dot_product_attention if using the dummy func
            # (batch, seq, head, dim) -> (batch, head, seq, dim)
            q_attn = q.unsqueeze(0).transpose(1, 2)
            k_attn = k.unsqueeze(0).transpose(1, 2)
            v_attn = v.unsqueeze(0).transpose(1, 2)
            
            # Note: The dummy function is called here.
            # The original `flash_attn_varlen_func_v3` expects a different shape.
            logger.info(f"q_attn shape: {q_attn.shape}, k_attn shape: {k_attn.shape}, v_attn shape: {v_attn.shape}, cu_seqlens_qkv: {cu_seqlens_qkv}, max_seqlen_qkv: {max_seqlen_qkv}")
            attn = flash_attn_varlen_func_v3(
                q_attn, k_attn, v_attn,
                None, None, None, None
            ).transpose(1, 2).squeeze(0).reshape(max_seqlen_qkv, -1)


        with record_function("ulysses/split_attn"):
            img_attn, txt_attn = attn[:img_q.shape[0]], attn[img_q.shape[0]:]

        with record_function("ulysses/all_gather_txt"):
            gathered_txt_attn = [torch.empty_like(txt_attn) for _ in range(world_size)]
            dist.all_gather(gathered_txt_attn, txt_attn, group=seq_p_group)

        with record_function("ulysses/all2all_head2seq"):
            img_attn = img_attn.reshape(-1, shard_heads, hidden_dims) # Reshape adapted for dummy
            img_attn = all2all_head2seq(img_attn, group=seq_p_group)
            img_attn = img_attn.reshape(shard_seqlen, -1)
            torch.cuda.synchronize()

        with record_function("ulysses/final_concat"):
            txt_attn = torch.cat(gathered_txt_attn, dim=1)
            attn = torch.cat([img_attn, txt_attn], dim=0)

        return attn



# --- Test Runner Function (Modified slightly for clarity) ---
def run_apply():
    # Get rank and world size AFTER initialization
    world_size = dist.get_world_size()
    rank = dist.get_rank()
 
    total_seqlen = 675 * 8
    nheads = 40
    hdim = 128
    
    # Important: Set device for the current process
    torch.cuda.set_device(rank)
    local_device = torch.device(f"cuda:{rank}")

    # Your logic for splitting sequence length was incorrect.
    # This part assumes you want to split image tokens across GPUs.
    # In Ulysses, often image tokens are sequence parallel and text tokens are head parallel.
    # For this example, we assume `total_seqlen` is the number of image tokens.
    assert total_seqlen % world_size == 0, "Total seqlen must be divisible by world size for this simple split"
    local_seqlen = total_seqlen // world_size

    q = torch.randn(local_seqlen, nheads, hdim, device=local_device, dtype=torch.bfloat16)
    k = torch.randn(local_seqlen, nheads, hdim, device=local_device, dtype=torch.bfloat16)
    v = torch.randn(local_seqlen, nheads, hdim, device=local_device, dtype=torch.bfloat16)

    # In this simplified example, all tokens are "image" tokens for sequence parallelism
    img_qkv_len = local_seqlen
    cu_seqlens_qkv = torch.tensor([0, local_seqlen], dtype=torch.int32, device=local_device)

    attn_weight = UlyssesAttnWeight()

    logger.info(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}, img_qkv_len: {img_qkv_len}, cu_seqlens_qkv: {cu_seqlens_qkv}")

    for _ in range(3):# Warm-up runs
        out = attn_weight.apply(
        q, k, v,
        img_qkv_len=img_qkv_len,
        cu_seqlens_qkv=cu_seqlens_qkv,
        attention_module="attention_module",
        seq_p_group=dist.group.WORLD, # Using the default group
        model_cls=None
    )
    torch.cuda.synchronize()

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        for _ in range(10):
            out = attn_weight.apply(
                q, k, v,
                img_qkv_len=img_qkv_len,
                cu_seqlens_qkv=cu_seqlens_qkv,
                attention_module="attention_module",
                seq_p_group=dist.group.WORLD, # Using the default group
                model_cls=None
            )
        prof.step()
    if rank == 0:
        print(prof.key_averages(group_by_stack_n=10).table(sort_by="cuda_time_total", row_limit=20))
        prof.export_chrome_trace("trace_main_block.json")
    return out

if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
 
    run_apply()

    dist.destroy_process_group()
