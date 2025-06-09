# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
from numpy import dtype
import triton
import triton.language as tl

@triton.jit
def mse(predictions_ptr, targets_ptr, mse_ptr, N, BLOCK_SIZE: tl.constexpr):
    predictions_ptr = predictions_ptr.to(tl.pointer_type(tl.float32))
    targets_ptr = targets_ptr.to(tl.pointer_type(tl.float32))
    mse_ptr = mse_ptr.to(tl.pointer_type(tl.float32))

    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0,BLOCK_SIZE)
    mask = idx < N 
    x = tl.load(predictions_ptr + idx,mask = mask,other = 0.0)
    y = tl.load(targets_ptr + idx,mask = mask,other = 0.0)
    value = y - x
    value = value * value / N 
    value = tl.sum(value)
    tl.atomic_add(mse_ptr,value)

# predictions_ptr, targets_ptr, mse_ptr are raw device pointers
def solve(predictions_ptr: int, targets_ptr: int, mse_ptr: int, N: int):
    BLOCK_SIZE = 1024
    num_blocks = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    mse[(num_blocks,)](
        predictions_ptr, targets_ptr, mse_ptr, N, BLOCK_SIZE,
    )


def main():
    import torch
    import torch.nn.functional as F     
    N = 8293
    predictions = torch.randn(N, dtype=torch.float32).to("cuda")
    targets = torch.randn(N, dtype=torch.float32).to("cuda")
    mse = torch.zeros(1, dtype=torch.float32, device='cuda')

    solve(
        int(predictions.data_ptr()),
        int(targets.data_ptr()),
        int(mse.data_ptr()),
        N
    )

    mse = mse.cpu().item()
    expected = F.mse_loss(predictions, targets, reduction='mean').cpu()
    
    print(predictions)
    print(targets)
    print(f"Triton Output: {mse:.6f}")
    print(f"Expected Sum:  {expected:.6f}")
    print(f"Difference:    {abs(mse - expected):.6f}")

if __name__ == "__main__":
    main()