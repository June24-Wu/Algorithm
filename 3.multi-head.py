import torch
import torch.nn.functional as F


class Solution:
    def __init__(self, embed_dim, num_heads):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        self.W_q = torch.nn.Linear(embed_dim, embed_dim)
        self.W_k = torch.nn.Linear(embed_dim, embed_dim)
        self.W_v = torch.nn.Linear(embed_dim, embed_dim)
        self.W_o = torch.nn.Linear(embed_dim, embed_dim)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            torch.nn.init.eye_(layer.weight)
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)

    def causal_attention(self, q, k, v, mask=None, casual=True):
        """
        q = [N, L, D]
        k = [N, L, D]
        v = [N, L, D]
        return: [N,L,D]
        """
        N, L, D = q.shape

        # linear
        # D' -> head_dim , H -> num_heads
        # [N,L,D] -> [N,L,H,D'] -> [N,H,L,D']
        # q = self.W_q(q).view(N,L,self.num_heads,self.head_dim).transpose(1,2) # [N,H,L,D']
        # k = self.W_k(k).view(N,L,self.num_heads,self.head_dim).transpose(1,2) # [N,H,L,D']
        # v = self.W_v(v).view(N,L,self.num_heads,self.head_dim).transpose(1,2) # [N,H,L,D']
        q = q.view(N, L, self.num_heads, self.head_dim).transpose(
            1, 2)  # [N,H,L,D']
        k = k.view(N, L, self.num_heads, self.head_dim).transpose(
            1, 2)  # [N,H,L,D']
        v = v.view(N, L, self.num_heads, self.head_dim).transpose(
            1, 2)  # [N,H,L,D']

        # q @ k.T / d_k
        d_k = torch.tensor(self.head_dim, dtype=torch.float32)  # D
        # [N,H,L,D'] @ [N,H,D',L] -> [N,H,L,L]
        score = q @ k.transpose(-2, -1) / torch.sqrt(d_k)

        # apply casual mask
        if casual:
            casual_mask = torch.tril(torch.ones(L, L, device=q.device)).unsqueeze(
                0).unsqueeze(0)  # [1,1,L,L]
            mask = casual_mask * mask if mask is not None else casual_mask

        # apply mask
        score = score.masked_fill(
            mask == 0, -1e9) if mask is not None else score  # [N,H,L,L]

        # softmax
        weight = F.softmax(score, dim=-1)  # [N,H,L,L]
        out = weight @ v  # [N,H,L,L] @ [N,H,L,D'] -> [N,H,L,D']
        out = out.transpose(1, 2).contiguous().view(
            N, L, self.embed_dim)  # [N,H,L,D'] -> [N,L,H,D'] -> [N,L,D]
        # out = self.W_o(out) # [N,L,D]
        return out, weight


def test_case_1(sol):
    print("Test Case 1: Shape check")
    q = k = v = torch.randn(2, 5, 64)  # batch=2, seq_len=5, dim=64
    out, attn = sol.causal_attention(q, k, v)
    assert out.shape == (2, 5, 64), "Output shape incorrect"
    assert attn.shape == (2, sol.num_heads, 5, 5), "Attention shape incorrect"
    print("✅ Passed!")


def test_case_2(sol):
    print("Test Case 2: Causal mask blocks future tokens")
    q = k = v = torch.ones(1, 4, 64)
    out, attn = sol.causal_attention(q, k, v)
    upper_triangle = torch.triu(attn[0, 0], diagonal=1)
    assert torch.allclose(upper_triangle, torch.zeros_like(
        upper_triangle), atol=1e-4), "Future tokens not masked"
    print("✅ Passed!")


def test_case_3(sol):
    print("Test Case 3: Output reflects token values under causal mask")
    v = torch.tensor([[[1.]*64, [2.]*64, [3.]*64, [4.]*64]]
                     )  # increasing values
    q = k = v.clone()
    out, attn = sol.causal_attention(q, k, v)
    token = out[0].mean(dim=[1])
    print(token)
    assert torch.allclose(token[0], torch.tensor([1.0]), atol=1e-2)
    assert torch.allclose(token[1], torch.tensor([2.0]), atol=1e-2)
    assert torch.allclose(token[2], torch.tensor([3.0]), atol=1e-2)
    assert torch.allclose(token[3], torch.tensor([4.0]), atol=1e-2)
    print("✅ Passed!")


def test_case_4(sol):
    print("Test Case 4: Works with mask + causal")
    q = k = v = torch.randn(1, 3, 64)
    mask = torch.tensor([[[1, 1, 0],
                          [1, 1, 0],
                          [1, 1, 1]]], dtype=torch.float32).unsqueeze(1)  # [1, 1, 3, 3]
    out, attn = sol.causal_attention(q, k, v, mask=mask)
    assert torch.allclose(attn[0, 0, 1, 2], torch.tensor(
        0.0)), "Masked token should not be attended"
    print("✅ Passed!")


def test_case_5(sol):
    print("Test Case 5: Multi-head consistency check")
    q = k = v = torch.randn(1, 6, 64)
    out, attn = sol.causal_attention(q, k, v)
    assert out.shape == (1, 6, 64), "Output shape incorrect for multi-head"
    print("✅ Passed!")


if __name__ == "__main__":
    sol = Solution(embed_dim=64, num_heads=8)
    test_case_1(sol)
    test_case_2(sol)
    test_case_3(sol)
    test_case_4(sol)
    test_case_5(sol)
    print("✅ All test cases passed!")
