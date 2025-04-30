import torch
import torch.nn.functional as F


class Solution:
    def causal_attention(self, q, k, v, mask=None, casual=True):
        """
        q = [N, H, L, D]
        k = [N, H, S, D]
        v = [N, H, S, D]
        return: [N,H,L,D]
        """
        L = q.shape[-2]
        S = k.shape[-2]
        d_k = torch.tensor(q.shape[-1], dtype=torch.float32)  # D
        score = q @ k.transpose(-2, -1) / torch.sqrt(d_k)  # [N,H,L,S]
        if casual:
            casual_mask = torch.tril(torch.ones(
                L, S, device=q.device)).unsqueeze(0).unsqueeze(0)
            if mask is not None:
                mask = casual_mask * mask
            else:
                mask = casual_mask

        if mask is not None:
            score = score.masked_fill(mask == 0, -1e9)
        weight = F.softmax(score, dim=-1)
        out = weight @ v
        return out, weight


def test_case_1(sol):
    print("Running Test Case 1")
    Q = K = V = torch.tensor([[[[1.], [2.], [3.]]]])  # Shape: [1, 1, 3, 1]
    out, attn = sol.causal_attention(Q, K, V)

    # Test if the attention weights for future tokens are zeroed out
    assert torch.all(attn[0, 0, 0, 1:] ==
                     0), "Token 0 should not attend to future tokens"
    assert torch.all(attn[0, 0, 1, 2:] ==
                     0), "Token 1 should not attend to future tokens"
    assert torch.all(attn[0, 0, 2, :] >
                     0), "Token 2 should attend to all tokens"

    print("✅ Test Case 1 passed!")


def test_case_2(sol):
    print("Running Test Case 2")
    Q = K = V = torch.tensor(
        [[[[1., 1.], [2., 2.], [3., 3.]]]])  # Shape: [1, 1, 3, 2]
    out, attn = sol.causal_attention(Q, K, V)

    # Verifying the shape
    assert out.shape == (1, 1, 3, 2), "Output shape is incorrect!"

    # Test if future attention is masked
    assert torch.all(attn[0, 0, 0, 1:] ==
                     0), "Token 0 should not attend to future tokens"
    assert torch.all(attn[0, 0, 1, 2:] ==
                     0), "Token 1 should not attend to future tokens"
    assert torch.all(attn[0, 0, 2, :] >
                     0), "Token 2 should attend to all tokens"

    print("✅ Test Case 2 passed!")


def test_case_3(sol):
    print("Running Test Case 3")
    Q = K = V = torch.tensor(
        [[[[1., 1.], [2., 2.], [3., 3.], [4., 4.]]]])  # Shape: [1, 1, 4, 2]
    out, attn = sol.causal_attention(Q, K, V)

    # Verifying the shape
    assert out.shape == (1, 1, 4, 2), "Output shape is incorrect!"

    # Test if future attention is masked
    assert torch.all(attn[0, 0, 0, 1:] ==
                     0), "Token 0 should not attend to future tokens"
    assert torch.all(attn[0, 0, 1, 2:] ==
                     0), "Token 1 should not attend to future tokens"
    assert torch.all(attn[0, 0, 2, 3:] ==
                     0), "Token 2 should not attend to future tokens"
    assert torch.all(attn[0, 0, 3, :] >
                     0), "Token 3 should attend to all tokens"

    print("✅ Test Case 3 passed!")


def test_case_4(sol):
    print("Running Test Case 4")
    Q = K = V = torch.tensor([[[[0.5], [0.6], [0.7]]]])  # Shape: [1, 1, 3, 1]
    out, attn = sol.causal_attention(Q, K, V)

    # Now, we should calculate the expected results based on softmax and weighted sums
    # Token 0 should only depend on itself, so it should be 0.5
    assert torch.allclose(out[0, 0, 0], torch.tensor(
        [0.5]), atol=1e-4), "Token 0 output is incorrect"

    # Token 1 should be a weighted sum of Token 0 and Token 1 based on softmax
    # Expecting a value closer to 0.55, as softmax gives weight to both 0.5 and 0.6
    # Printing actual value for debugging
    print(f"Token 1 output: {out[0, 0, 1]}")
    assert torch.allclose(out[0, 0, 1], torch.tensor(
        [0.55]), atol=1e-2), "Token 1 output is incorrect"

    # Token 2 should be a weighted sum of Token 0, Token 1, and Token 2 based on softmax
    # Expecting a value closer to 0.6, as softmax gives weight to 0.5, 0.6, and 0.7
    # Printing actual value for debugging
    print(f"Token 2 output: {out[0, 0, 2]}")
    assert torch.allclose(out[0, 0, 2], torch.tensor(
        [0.6]), atol=1e-2), "Token 2 output is incorrect"

    print("✅ Test Case 4 passed!")


def test_case_5(sol):
    print("Running Test Case 5")
    Q = torch.randn(1, 1, 3, 4)  # Random queries
    K = torch.randn(1, 1, 3, 4)  # Random keys
    V = torch.randn(1, 1, 3, 4)  # Random values
    out, attn = sol.causal_attention(Q, K, V)

    # Verifying the shape
    assert out.shape == (1, 1, 3, 4), "Output shape is incorrect!"

    # Test if future attention is masked
    assert torch.all(attn[0, 0, 0, 1:] ==
                     0), "Token 0 should not attend to future tokens"
    assert torch.all(attn[0, 0, 1, 2:] ==
                     0), "Token 1 should not attend to future tokens"
    assert torch.all(attn[0, 0, 2, :] >
                     0), "Token 2 should attend to all tokens"

    print("✅ Test Case 5 passed!")


def test_case_6(sol):
    print("Running Test Case 6")

    base = torch.tensor([[1.0, 2.0, 3.0, 4.0]])  # [1, 4]
    Q = K = V = base.unsqueeze(0).unsqueeze(
        0).transpose(-2, -1).repeat(1, 1, 1, 4)  # [1, 1, 4, 4]

    out, attn = sol.causal_attention(Q, K, V)
    assert out.shape == (1, 1, 4, 4), "Output shape is incorrect!"

    # 验证注意力mask
    assert torch.all(attn[0, 0, 0, 1:] ==
                     0), "Token 0 should not attend to future tokens"
    assert torch.all(attn[0, 0, 1, 2:] ==
                     0), "Token 1 should not attend to future tokens"
    assert torch.all(attn[0, 0, 3, :] >
                     0), "Token 3 should attend to all tokens"

    print("✅ Test Case 6 passed!")


if __name__ == "__main__":
    sol = Solution()
    test_case_1(sol)
    test_case_2(sol)
    test_case_3(sol)
    test_case_4(sol)
    test_case_5(sol)
    test_case_6(sol)
    print("✅ All test cases passed!")
