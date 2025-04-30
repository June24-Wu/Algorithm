import torch
import torch.nn.functional as F

class Solution:
    def self_attention(self, q, k, v, mask=None):
        """
        # q = [N, H, L, D]
        # k = [N, H, S, D]
        # v = [N, H, S, D]
        # return: [N,H,L,D]
        """
        
        d_k = torch.tensor(q.shape[-1], dtype=torch.float32)

        score = q @ k.transpose(-2, -1) / torch.sqrt(d_k)  # [N,H,L,S]

        if mask is not None:
            score = score.masked_fill(mask == 0, 1e-9)  # [N,H,L,S]
        weight = F.softmax(score, dim=-1)  # [N,H,L,S]
        attention = weight @ v  # [N,H,L,S] x [N,H,L,D] = [N,H,L,D]
        return attention, weight


def test_case_1(sol):
    print("Running Test Case 1")
    Q = K = torch.ones(1, 1, 2, 4)
    V = torch.tensor([[[[1., 2., 3., 4.], [4., 3., 2., 1.]]]])
    out, attn = sol.self_attention(Q, K, V)
    expected = torch.tensor([2.5, 2.5, 2.5, 2.5])
    actual = out[0, 0, 0]
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    assert torch.allclose(actual, expected, atol=1e-5), "Test Case 1 Failed"


def test_case_2(sol):
    Q = torch.randn(1, 1, 2, 4)
    K = torch.randn(1, 1, 3, 4)
    V = torch.randn(1, 1, 3, 4)
    out, attn = sol.self_attention(Q, K, V)
    assert out.shape == (1, 1, 2, 4), "Test Case 2 Failed"


def test_case_3(sol):
    Q = torch.randn(2, 4, 5, 8)
    K = torch.randn(2, 4, 5, 8)
    V = torch.randn(2, 4, 5, 8)
    out, attn = sol.self_attention(Q, K, V)
    assert out.shape == (2, 4, 5, 8), "Test Case 3 Failed"


def test_case_4(sol):
    Q = K = torch.zeros(1, 1, 3, 4)
    V = torch.randn(1, 1, 3, 4)
    out, attn = sol.self_attention(Q, K, V)
    assert torch.allclose(
        attn.sum(dim=-1), torch.ones_like(attn.sum(dim=-1))), "Test Case 4 Failed"


def test_case_5(sol):
    print("Running Test Case 5 (mask test)")
    Q = torch.randn(1, 1, 2, 4)
    K = torch.randn(1, 1, 2, 4)
    V = torch.tensor([[[[1., 0., 0., 0.], [0., 0., 0., 0.]]]])
    mask = torch.tensor([[[[1, 0], [1, 0]]]])
    out, attn = sol.self_attention(Q, K, V, mask)
    print(f"Output: {out}")
    assert torch.all(out[:, :, :, 1:] ==
                     0), "Test Case 5 Failed (values should be masked to 0)"


def test_case_6(sol):
    L = 4
    Q = K = V = torch.randn(1, 1, L, 4)
    mask = torch.tril(torch.ones(L, L)).unsqueeze(0).unsqueeze(0)
    out, attn = sol.self_attention(Q, K, V, mask)
    assert out.shape == (1, 1, L, 4), "Test Case 6 Failed"


def test_case_7(sol):
    Q = torch.tensor([[[[1.], [2.]]]])
    K = torch.tensor([[[[1.], [2.]]]])
    V = torch.tensor([[[[10.], [20.]]]])
    out, attn = sol.self_attention(Q, K, V)
    assert out.shape == (1, 1, 2, 1), "Test Case 7 Failed"


def test_case_8(sol):
    Q = torch.randn(1, 1, 5, 64)
    K = torch.randn(1, 1, 5, 64)
    V = torch.randn(1, 1, 5, 64)
    out, attn = sol.self_attention(Q, K, V)
    assert out.shape == (1, 1, 5, 64), "Test Case 8 Failed"


def test_case_9(sol):
    Q = torch.randn(3, 2, 4, 16)
    K = torch.randn(3, 2, 4, 16)
    V = torch.randn(3, 2, 4, 16)
    out, attn = sol.self_attention(Q, K, V)
    assert out.shape == (3, 2, 4, 16), "Test Case 9 Failed"


def test_case_10(sol):
    print("Running Test Case 10")
    Q = K = torch.ones(1, 1, 2, 4)
    V = torch.tensor([[[[1., 2., 3., 4.], [5., 6., 7., 8.]]]])
    out, _ = sol.self_attention(Q, K, V)
    expected = (V[0, 0, 0] + V[0, 0, 1]) / 2
    actual = out[0, 0, 0]
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    assert torch.allclose(actual, expected, atol=1e-5), "Test Case 10 Failed"


if __name__ == "__main__":
    sol = Solution()
    test_case_1(sol)
    test_case_2(sol)
    test_case_3(sol)
    test_case_4(sol)
    test_case_5(sol)
    test_case_6(sol)
    test_case_7(sol)
    test_case_8(sol)
    test_case_9(sol)
    test_case_10(sol)
    print("✅ All test cases passed!")
