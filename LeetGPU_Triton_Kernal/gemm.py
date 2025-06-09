# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def matmal(a_ptr, b_ptr, c_ptr, M, N, K: tl.constexpr, alpha, beta):

    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))

    row_id = tl.program_id(0)
    col_id = tl.program_id(1)

    if row_id >= M or col_id >= N:
        return

    c_value = tl.load(c_ptr + row_id * N + col_id)
    c_value = beta * c_value

    result = 0.0
    for k in range(0, K):
        a_idx = row_id * K + k
        b_idx = k * N + col_id

        a_val = tl.load(a_ptr + a_idx)
        b_val = tl.load(b_ptr + b_idx)
        result += a_val * b_val
    c_value += alpha * result
    c_value = c_value.to(tl.float16)
    tl.store(c_ptr + row_id * N + col_id,c_value)



def solve(a_ptr: int, b_ptr: int, c_ptr: int, M: int, N: int, K: int, alpha: float, beta: float):
    matmal[(M,K)](
        a_ptr, b_ptr, c_ptr, M, N, K, alpha, beta
    )


def main():
    import torch

    M, N, K = 2, 2, 3
    alpha = 1.0
    beta = 0.0


    A = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float32, device='cuda')
    B = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], dtype=torch.float32, device='cuda')
    C = torch.ones((M, N), dtype=torch.float32, device='cuda')

    print("Input A:\n", A.cpu().numpy())
    print("Input B:\n", B.cpu().numpy())
    print("Initial C:\n", C.cpu().numpy())

    solve(
        A.data_ptr(),
        B.data_ptr(),
        C.data_ptr(),
        M, N, K,
        alpha, beta
    )

    torch.cuda.synchronize()
    print("Output C:\n", C.cpu().numpy())

if __name__ == "__main__":
    main()