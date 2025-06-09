import triton
import triton.language as tl

@triton.jit
def add_kernal(y_ptr, result_ptr, n, scale, BLOCK_SIZE: tl.constexpr):
    y_ptr = y_ptr.to(tl.pointer_type(tl.float32))
    result_ptr = result_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    x = tl.load(y_ptr + idx)

    x = tl.sum(x)
    tl.atomic_add(result_ptr,x)

@triton.jit
def reduce_kernal(y_ptr, result_ptr, n, scale, BLOCK_SIZE: tl.constexpr):
    y_ptr = y_ptr.to(tl.pointer_type(tl.float32))
    result_ptr = result_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    
    x = tl.load(result_ptr + pid)
    x = scale * x
    tl.store(result_ptr,x)


def solve(y_samples_ptr: int, result_ptr: int, a: float, b: float, n_samples: int):
    BLOCK_SIZE = 1024
    num_blocks = (n_samples + BLOCK_SIZE - 1) // BLOCK_SIZE
    scale = (b - a) / n_samples
    add_kernal[(num_blocks,)](
        y_samples_ptr, result_ptr, n_samples, scale, BLOCK_SIZE
    )

    reduce_kernal[(1,)](
        y_samples_ptr, result_ptr, n_samples, scale, BLOCK_SIZE
    )

def main():
    import torch
    a = 0.0
    b = 2.0
    n_samples = 4096
    y_samples = torch.randn(n_samples, dtype=torch.float32, device='cuda')
    result = torch.zeros(1, dtype=torch.float32, device='cuda')

    solve(y_samples.data_ptr(), result.data_ptr(), a, b, n_samples)
    torch.cuda.synchronize()

    expect = (b-a) / n_samples * torch.sum(y_samples)
    output_cpu = result.cpu().item()
    expected = expect.sum().item()

    print(f"Triton Output: {output_cpu:.6f}")
    print(f"Expected Sum:  {expected:.6f}")
    print(f"Difference:    {abs(output_cpu - expected):.6f}")

if __name__ == "__main__":
    main()