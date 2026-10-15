"""Implement causal Grouped Query Attention in the student section below."""

from __future__ import annotations

import numpy as np


class GroupQueryAttention:
    """Causal grouped-query self-attention with a compact KV cache."""

    def __init__(
        self,
        d_model: int,
        num_query_heads: int,
        num_kv_heads: int,
        seed: int = 0,
    ) -> None:
        if d_model <= 0 or num_query_heads <= 0 or num_kv_heads <= 0:
            raise ValueError("Dimensions and head counts must be positive")
        if d_model % num_query_heads != 0:
            raise ValueError("d_model must be divisible by num_query_heads")
        if num_query_heads % num_kv_heads != 0:
            raise ValueError("num_query_heads must be divisible by num_kv_heads")

        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_query_heads
        self.group_size = num_query_heads // num_kv_heads

        rng = np.random.default_rng(seed)
        scale = 1.0 / np.sqrt(d_model)
        self.Wq = rng.normal(
            0.0, scale, size=(d_model, num_query_heads * self.head_dim)
        )
        self.Wk = rng.normal(
            0.0, scale, size=(d_model, num_kv_heads * self.head_dim)
        )
        self.Wv = rng.normal(
            0.0, scale, size=(d_model, num_kv_heads * self.head_dim)
        )
        self.Wo = rng.normal(
            0.0, scale, size=(num_query_heads * self.head_dim, d_model)
        )

    def forward(
        self,
        x: np.ndarray,
        cache: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray]]:
        """Return new-token outputs and the updated compact ``(K, V)`` cache."""
        # ============== Student section: implement everything here ==============
        B, S, D = x.shape
        q = (x @ self.Wq).reshape(B,S,self.num_query_heads,self.head_dim).transpose(0,2,1,3) # (B,S,D) @ (D,H_q * D') -> (B,S,H_q * D') -> (B,S,H_q,D') -> (B,H_q,S,D')
        k = (x @ self.Wk).reshape(B,S,self.num_kv_heads,self.head_dim).transpose(0,2,1,3) # (B,S,D) @ (D,H_kv * D') -> (B,S,H_kv * D') -> (B,S,H_kv,D') -> (B,H_kv,S,D')
        v = (x @ self.Wv).reshape(B,S,self.num_kv_heads,self.head_dim).transpose(0,2,1,3) # (B,S,D) @ (D,H_kv * D') -> (B,S,H_kv * D') -> (B,S,H_kv,D') -> (B,H_kv,S,D')

        if cache is None:
            past_length = 0
            all_k, all_v = k, v
        else:
            past_k, past_v = cache
            past_length = past_k.shape[2]
            all_k = np.concatenate([past_k,k],axis=2)
            all_v = np.concatenate([past_v,v],axis=2)

        # Repeat K, V head. DO NOT USE np.concatenate([all_k] * group_size,axis=1)
        # Because it will convert Head to [K0, K1, K0, K1, K0, K1, K0, K1]. 
        # But the GQA will be [K0, K0, K0, K0, K1, K1, K1, K1]
        all_k_exp = np.repeat(all_k,self.group_size,axis=1) # (B,H_q,S_all,D')
        all_v_exp = np.repeat(all_v,self.group_size,axis=1) # (B,H_q,S_all,D')

        # q @ k.T / sqrt(head_dim)
        scores = q @ all_k_exp.transpose(0,1,3,2) / np.sqrt(self.head_dim) # (B,H_q,S_all,D') @ (B,H_q,D',S) -> (B,H_q,S,S)

        casual_mask = np.tril(np.ones([S,S],dtype=bool),k=0)
        past_mask = np.ones([S,past_length],dtype=bool)
        casual_mask = np.concatenate([past_mask,casual_mask],axis=1)

        def softmax(x):
            x = np.exp(x - np.max(x,axis=-1,keepdims=True))
            return x / np.sum(x,axis=-1,keepdims=True)

        scores = np.where(casual_mask,scores,-np.inf)
        weight = softmax(scores) # (B,H_q, S, S_all)

        output = (weight @ all_v_exp).transpose(0,2,1,3).reshape(B,S,-1) # (B,H_q, S, S_all) @ (B,H_q, S_all,D') ->  (B,H_q, S, D') -> (B,S,H_q,D') -> (B,S,H_q * D')
        output = output @ self.Wo
        return output , (all_k,all_v)

        # ============ End student section: implement everything here ============
