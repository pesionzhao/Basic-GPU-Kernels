import torch
import triton
import triton.language as tl
import my_cuda_ops
# 如果向量长度太长，需要进行分块循环
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
    ],
    key=['n_elements']
)
@triton.jit
def reduce_sum_kernel(
        input_ptr,  # 第一个向量的指针
        output_ptr,  # 输出向量的指针
        n_elements,  # 向量的大小
        BLOCK_SIZE: tl.constexpr,  # 每个 block 的大小
):
    # 一行的元素数量 N 一般远超 BLOCK_SIZE，故需要对 N 进行分块计算
    sum = 0
    _sum= tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in tl.range(0, n_elements, BLOCK_SIZE):
        # 创建一个掩码以防止越界访问
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        # 从全局内存加载数据
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
        _sum += x
        # 将结果存储回全局内存
    sum = tl.sum(_sum, axis=0)
    if tl.program_id(0) == 0:
        tl.store(output_ptr, sum)
    # tl.store(output_ptr + offsets, output, mask=mask)

def reduce_sum(x: torch.Tensor):
    assert x.is_cuda
    n_elements = x.numel()
    # BLOCK_SIZE = triton.next_power_of_2(n_elements)

    # 输出只有一个元素
    output = torch.empty(1, device='cuda', dtype=torch.float32)

    # 启动 grid, 这里只用一个线程块
    reduce_sum_kernel[(1,)](x, output, n_elements)

    return output.item()



if __name__ == '__main__':
    numel = 4096
    b = torch.empty(1, device='cuda', dtype=torch.float32)
    a = torch.rand(numel,device='cuda', dtype=torch.float32)
    output = reduce_sum(a)
    output = torch.empty(1, device='cuda', dtype=torch.float32)
    print(a)
    my_cuda_ops.sum(a,b,numel,1)
    print(torch.sum(a))
    print(b)
    print(output)

