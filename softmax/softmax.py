import torch
import triton
import triton.language as tl
import my_cuda_ops

# 本质就是一个block计算一行，
@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    row_len = 2 # 每个内核循环处理两行数据
    row_start = tl.program_id(0) * row_len
    if row_start >= n_rows: # 行的mask
        return
    for row_idx in tl.range(row_start, row_start + row_len, 1):
        # The stride represents how much we need to increase the pointer to advance 1 row
        # 步长表示我们需要对指针增加多少以推进 1 行
        row_start_ptr = input_ptr + row_idx * input_row_stride
        # The block size is the next power of two greater than n_cols, so we can fit each
        # 块大小是大于 n_cols 的下一个二的幂，因此我们可以适配
        # row in a single block
        # 单个块中的行
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = row_start_ptr + col_offsets
        # Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
        # 将行加载到 SRAM 中，使用掩码，因为 BLOCK_SIZE 可能大于 n_cols
        mask = col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        # Subtract maximum for numerical stability
        # 为了数值稳定性而减去最大值
        row_minus_max = row - tl.max(row, axis=0)
        # Note that exponentiation in Triton is fast but approximate (i.e., think __expf in CUDA)
        # 请注意，Triton 中的指数运算速度很快，但是是近似的（例如，类似于 CUDA 中的 __expf）。
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        # Write back output to DRAM
        # 将输出写回 DRAM
        output_row_start_ptr = output_ptr + row_idx * output_row_stride
        output_ptrs = output_row_start_ptr + col_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)


input_tensor = torch.randn(1024, 512, device='cuda')  # 1000x512 的随机张量
output_tensor = torch.empty_like(input_tensor)  # 用于存储输出的张量

# 2. 定义 kernel 的网格和块大小
n_rows, n_cols = input_tensor.shape
BLOCK_SIZE = triton.next_power_of_2(n_cols)  # 计算大于 n_cols 的下一个 2 的幂
num_stages = 3  # 可以根据硬件调整

# 3. 调用 kernel
grid = lambda meta: (triton.cdiv(n_rows, 2),)  # 网格大小为行数
softmax_kernel[grid](
    output_tensor, input_tensor,
    input_tensor.stride(0), output_tensor.stride(0),  # 行步长
    n_rows, n_cols,
    BLOCK_SIZE=BLOCK_SIZE,
)

# 4. 验证结果
# 使用 PyTorch 的 softmax 作为参考
expected_output = torch.softmax(input_tensor, dim=1)
cuda_output = torch.empty_like(input_tensor)  # 用于存储输出的张量
# my_cuda_ops.softmax(input_tensor, cuda_output, 1024, 512)
print("Triton Softmax 和 PyTorch Softmax 是否接近:", torch.allclose(cuda_output, expected_output, atol=1e-6))