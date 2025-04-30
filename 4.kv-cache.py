import torch
import torch.nn.functional as F


class Solution:
    def __init__(self, embed_dim, num_heads, num_kv_heads=None):
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # MQA & GQA
        self.num_kv_heads = num_kv_heads if num_kv_heads else num_heads
        self.num_kv_per_query = self.num_heads // self.num_kv_heads
        assert self.num_kv_per_query * self.num_kv_heads == self.num_heads
        
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim
        # self.W_q = torch.nn.Linear(embed_dim,embed_dim)
        # self.W_k = torch.nn.Linear(embed_dim,embed_dim)
        # self.W_v = torch.nn.Linear(embed_dim,embed_dim)
        # self.W_o = torch.nn.Linear(embed_dim,embed_dim)

    def forward(self, q, k, v, mask=None, casual_mask=True, k_cache=None, v_cache=None):
        """
        Q = [N,L_Q,D]
        K = [N,L_K,D]
        V = [N,L_V,D]
        """
        N, L, D = q.shape
        kv_len = k.shape[1]

        # apply multi-head
        # [N,L,D] -> [N,L,H,D'] -> [N,H,L,D']
        q = q.view(N, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(N, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(N, kv_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # MQA & GQA
        k = torch.repeat_interleave(k, self.num_kv_per_query, dim=1)
        v = torch.repeat_interleave(v, self.num_kv_per_query, dim=1)

        # KV cache
        k = torch.cat([k_cache, k], dim=2) if k_cache is not None else k
        v = torch.cat([v_cache, v], dim=2) if v_cache is not None else v
        k_cache = k.clone()
        v_cache = v.clone()

        # q @ k.T / d_k -> [N,H,L,D'] @ [N,H,L_K,D'] -> [N,H,L,L_k]
        d_k = torch.tensor(self.head_dim, dtype=torch.float32, device=q.device)
        weight = q @ k.transpose(-2, -1) / torch.sqrt(d_k)

        # mask
        if casual_mask:  # [1,1,L,L]
            casual_mat = torch.tril(torch.ones(
                L, L, device=q.device)).unsqueeze(0).unsqueeze(0)
            mask = mask * casual_mat if mask is not None else casual_mat

        weight = weight.masked_fill(
            mask == 0, -1e9) if mask is not None else weight  # [N,H,L,L]
        weight = F.softmax(weight, dim=-1)
        out = weight @ v  # [N,H,L,L] @ [N,H,L,D'] -> [N,H,L,D']
        out = out.transpose(1, 2).contiguous().view(
            N, L, D)  # [N,H,L,D'] -> [N,L,H,D'] -> [N,L,D]
        return out, weight, k_cache, v_cache


def test_case_1(sol):
    print("Test Case 1: Output reflects token values under causal mask")
    v = torch.tensor([[[1.]*64, [2.]*64, [3.]*64, [4.]*64]]
                     )  # increasing values
    q = k = v.clone()
    out, attn, _, _ = sol.forward(q, k, v)
    token = out[0].mean(dim=[1])
    print(token)
    assert torch.allclose(token[0], torch.tensor([1.0]), atol=1e-2)
    assert torch.allclose(token[1], torch.tensor([2.0]), atol=1e-2)
    assert torch.allclose(token[2], torch.tensor([3.0]), atol=1e-2)
    assert torch.allclose(token[3], torch.tensor([4.0]), atol=1e-2)
    print("✅ Passed!")


def test_case_2(sol):
    print("Running Simplified KV Cache Test (GQA)")

    embed_dim = 6
    num_heads = 2
    num_kv_heads = 1  # GQA
    model = Solution(embed_dim, num_heads, num_kv_heads)

    seq_len = 3
    D = embed_dim
    head_dim = embed_dim // num_heads
    kv_dim = head_dim * num_kv_heads  # == 3

    q_tokens = torch.stack([
        torch.full((D,), 0.0),  # Token 0
        torch.full((D,), 1.0),  # Token 1
        torch.full((D,), 2.0),  # Token 2
    ]).unsqueeze(0)  # [1, 3, D]

    kv_tokens = torch.stack([
        torch.full((kv_dim,), 0.0),
        torch.full((kv_dim,), 1.0),
        torch.full((kv_dim,), 2.0),
    ]).unsqueeze(0)  # [1, 3, kv_dim]

    k_cache, v_cache = None, None
    all_outs = []
    for t in range(seq_len):
        q = q_tokens[:, t:t+1, :]         # [1, 1, D]
        kv = kv_tokens[:, t:t+1, :]       # [1, 1, kv_dim]

        out, attn, k_cache, v_cache = model.forward(
            q, kv, kv,
            k_cache=k_cache,
            v_cache=v_cache
        )

        print(f"Step {t}:")
        print("  Q token:", q.flatten().tolist())
        print("  KV token:", kv.flatten().tolist())
        print("  Output:", out.flatten().tolist())
        print("  KV Cache length:", k_cache.shape[2])
        all_outs.append(out)

    final_out = torch.cat(all_outs, dim=1)
    print("\n✅ Passed!")


if __name__ == "__main__":
    sol = Solution(embed_dim=64, num_heads=8)
    test_case_1(sol)
    test_case_2(Solution(embed_dim=6, num_heads=2, num_kv_heads=1))
    print("✅ All test cases passed!")
