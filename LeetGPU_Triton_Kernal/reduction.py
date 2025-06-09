# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl


@triton.jit
def reduce_kernal(input_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)

    x = tl.load(input_ptr + offsets)
    x = tl.sum(x)
    tl.atomic_add(output_ptr, x)

def solve(input_ptr: int, output_ptr: int, N: int):
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    reduce_kernal[(num_blocks,)](input_ptr,output_ptr,N,BLOCK_SIZE)


def main():
    import torch
    N = 4096
    input_cpu = torch.randn(N, dtype=torch.float32)
    input_gpu = input_cpu.to('cuda')

    output_gpu = torch.zeros(1, dtype=torch.float32, device='cuda')

    solve(input_gpu.data_ptr(), output_gpu.data_ptr(), N)

    torch.cuda.synchronize()

    output_cpu = output_gpu.cpu().item()
    expected = input_cpu.sum().item()

    print(f"Triton Output: {output_cpu:.6f}")
    print(f"Expected Sum:  {expected:.6f}")
    print(f"Difference:    {abs(output_cpu - expected):.6f}")

if __name__ == "__main__":
    main()