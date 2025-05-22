import torch
import torch.nn as nn
import math

class SimpleRoPE(nn.Module):
    """
    A simplified implementation of Rotary Positional Embedding (RoPE).

    RoPE injects positional information into transformer models by rotating
    pairs of input features based on their absolute position. This version
    precomputes an embedding table of sines and cosines up to a maximum
    sequence length.

    Args:
        dim (int): The dimension of the features to which RoPE will be applied.
                   This is typically the head dimension in an attention mechanism,
                   or a fraction of it if only partial RoPE is used.
        max_position_embeddings (int, optional): The maximum sequence length
                                                 for which to precompute embeddings.
                                                 Defaults to 2048.
        base (float, optional): The base value used for calculating the
                                rotational frequencies (theta). Defaults to 10000.0.
    """

    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"Dimension must be an even number, but got {dim}")
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # theta_i = 1 / (base ** (2i / dim))
        theta_i = 1.0 / (self.base ** (torch.arange(0,dim,2,dtype = torch.float32) / dim))
        self.register_buffer("theta_i",theta_i,persistent=False)

        # freqs = t * theta_i -> [seq_len,1] @ [1,dim / 2] -> [seq_len, dim / 2]
        t = torch.arange(self.max_position_embeddings,dtype = torch.float32)
        freqs = t.unsqueeze(1) @ theta_i.unsqueeze(0)
        
        # emb = torch.cat((freqs,freqs),dim=-1) #  [seq_len,dim]
        emb = freqs.repeat_interleave(2,dim = 1)  #  [seq_len,dim // 2] ->  #  [seq_len,dim]

        self.register_buffer("cos_cached",emb.cos(),persistent=False)
        self.register_buffer("sin_cached",emb.sin(),persistent=False)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        Rotates half the hidden dimensions.
        For each input pair (x_i, x_{i+1}), it returns (-x_{i+1}, x_i).
        Equivalent to applying a 90-degree rotation in the 2D plane formed by each pair.

        Args:
            x (torch.Tensor): Input tensor, expected shape (..., dim).

        Returns:
            torch.Tensor: Tensor with rotated halves, same shape as x.
                          Example: if x is [x0, x1, x2, x3], output is [-x1, x0, -x3, x2].
        """
        x_reshaped = x.reshape(*x.shape[:-1], -1, 2)
        x1, x2 = x_reshaped[..., 0], x_reshaped[..., 1]
        return torch.stack((-x2, x1), dim=-1).reshape(*x.shape)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Apply RoPE to the input tensor x using the specified positions.

        Args:
            x (torch.Tensor): Input tensor. Expected shape: (..., seq_len, dim).
                              `dim` must match `self.dim`.
            positions (torch.Tensor): Tensor of token positions.
                                      Expected shape should be broadcastable to x.shape[:-1].
                                      Typically (seq_len,) or (batch_size, seq_len).
                                      Values should be in [0, max_position_embeddings - 1].

        Returns:
            torch.Tensor: Output tensor with RoPE applied, same shape as x.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Input tensor's last dimension ({x.shape[-1]}) "
                f"must match RoPE's configured dim ({self.dim})"
            )
        if positions.max() >= self.max_position_embeddings:
             raise ValueError(
                f"Position id {positions.max().item()} is out of range for RoPE cache "
                f"with max_len {self.max_position_embeddings}"
            )
        if positions.min() < 0:
            raise ValueError(f"Position id {positions.min().item()} is negative, not supported.")

        # Gather the cosine and sine values for the given positions.
        # self.cos_cached has shape (max_position_embeddings, dim)
        # positions shape e.g., (seq_len,) or (batch_size, seq_len)
        # cos/sin will have shape (*positions.shape, dim)
        # Convert to the dtype of x to avoid mixed precision issues.
        
        cos = self.cos_cached[positions].to(dtype=x.dtype)
        sin = self.sin_cached[positions].to(dtype=x.dtype)
        
        # The RoPE transformation:
        # For a complex number x_k = x_{2k} + i*x_{2k+1} and m*theta_k,
        # the rotated value is (x_{2k} + i*x_{2k+1}) * exp(i*m*theta_k)
        # = (x_{2k} + i*x_{2k+1}) * (cos(m*theta_k) + i*sin(m*theta_k))
        # = (x_{2k}*cos(m*theta_k) - x_{2k+1}*sin(m*theta_k))  (real part)
        #   + i*(x_{2k}*sin(m*theta_k) + x_{2k+1}*cos(m*theta_k)) (imaginary part)
        #
        # This can be implemented as: (x * cos) + (rotate_half(x) * sin)
        # where:
        #   x * cos gives:
        #     [x0*c0, x1*c0, x2*c1, x3*c1, ...] (c0=cos(m*th0), c1=cos(m*th1))
        #   rotate_half(x) gives:
        #     [-x1, x0, -x3, x2, ...]
        #   rotate_half(x) * sin gives:
        #     [-x1*s0, x0*s0, -x3*s1, x2*s1, ...] (s0=sin(m*th0), s1=sin(m*th1))
        # Summing these element-wise:
        #   Position 0: x0*c0 - x1*s0
        #   Position 1: x1*c0 + x0*s0
        #   Position 2: x2*c1 - x3*s1
        #   Position 3: x3*c1 + x2*s1
        # This matches the real and imaginary parts of the complex multiplication.
        # PyTorch broadcasting handles cases where cos/sin have fewer leading dimensions than x.
        
        if cos.dim() == 3 < x.dim() == 4:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        print("x.shape",x.shape)
        print("cos.shape",cos.shape)
        print("self._rotate_half(x).shape",self._rotate_half(x).shape)
        print("sin.shape",sin.shape)
        return (x * cos) + (self._rotate_half(x) * sin)

def test_simple_rope():
    # Hypothetical dimensions
    batch_size = 2
    num_heads = 4
    seq_len = 128
    head_dim = 8
    max_position_embeddings = 128

    # Fake input tensor
    x = torch.randn(batch_size, num_heads, seq_len, head_dim)

    # Position IDs: [batch, seq_len]
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

    # Your RoPE module
    rope = SimpleRoPE(dim=head_dim, max_position_embeddings=max_position_embeddings)  # example init
    rope.eval()

    with torch.no_grad():
        out = rope(x, position_ids)

    # Check shape
    assert out.shape == x.shape, f"Output shape mismatch: {out.shape} vs {x.shape}"

    # Check that values actually change (i.e., positional info applied)
    if torch.allclose(out, x):
        raise AssertionError("RoPE output is identical to input, rotation might not be applied.")

    print("✅ SimpleRoPE test passed!")

def test_position_0_identity():
    batch_size = 1
    num_heads = 1
    seq_len = 1
    head_dim = 4
    max_position_embeddings = 128

    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # shape [1,1,1,4]
    position_ids = torch.tensor([[0]])

    rope = SimpleRoPE(dim=head_dim, max_position_embeddings=max_position_embeddings)
    rope.eval()

    with torch.no_grad():
        out = rope(x, position_ids)

    assert torch.allclose(out, x, atol=1e-5), f"Output at position 0 should match input.\nGot: {out}, Expected: {x}"
    print("✅ test_position_0_identity passed!")


def test_manual_pi_half_rotation():
    class MockRoPE(SimpleRoPE):
        def get_cos_sin(self, position_ids):
            # Simulate 90-degree rotation: cos = 0, sin = 1
            B, L = position_ids.shape
            cos = torch.zeros(B, L, self.dim)
            sin = torch.ones(B, L, self.dim)
            return cos, sin

    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])  # [B=1, H=1, L=1, D=4]

    rope = MockRoPE(dim=4, max_position_embeddings=10)
    rope.eval()

    with torch.no_grad():
        out = rope._rotate_half(x)

    expected = torch.tensor([[[[-2.0, 1.0, -4.0, 3.0]]]])
    assert torch.allclose(out, expected, atol=1e-5), f"Expected: {expected}, Got: {out}"
    print("✅ test_manual_pi_half_rotation passed!")


def test_different_positions_change_output():
    batch_size = 1
    num_heads = 1
    seq_len = 2
    head_dim = 4
    max_position_embeddings = 128

    x = torch.randn(batch_size, num_heads, seq_len, head_dim)
    rope = SimpleRoPE(dim=head_dim, max_position_embeddings=max_position_embeddings)
    rope.eval()

    pos_ids_0 = torch.zeros((batch_size, seq_len), dtype=torch.long)
    pos_ids_1 = torch.ones((batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        out0 = rope(x, pos_ids_0)
        out1 = rope(x, pos_ids_1)

    assert not torch.allclose(out0, out1), "Outputs should differ across positions"
    print("✅ test_different_positions_change_output passed!")


def test_rope_correctness():
    dim = 4
    max_pos = 16
    rope = SimpleRoPE(dim=dim, max_position_embeddings=max_pos)
    
    # 构造一个简单的输入 (batch=1, seq_len=3, dim=4)
    x = torch.tensor([[[1., 2., 3., 4.],
                       [5., 6., 7., 8.],
                       [9., 10., 11., 12.]]])
    
    positions = torch.tensor([0, 1, 2])  # seq_len=3
    
    # 使用你的RoPE前向计算
    y = rope(x, positions)
    
    # 下面计算每个位置和pair手动旋转，验证结果
    theta_i = rope.theta_i  # 频率，shape (dim/2,)
    
    for pos_idx, pos in enumerate(positions):
        for pair_idx in range(dim // 2):
            theta = pos * theta_i[pair_idx]
            cos_theta = math.cos(theta.item())
            sin_theta = math.sin(theta.item())
            
            x0 = x[0, pos_idx, pair_idx*2].item()
            x1 = x[0, pos_idx, pair_idx*2 + 1].item()
            
            # 按RoPE旋转公式计算
            expected_0 = x0 * cos_theta - x1 * sin_theta
            expected_1 = x0 * sin_theta + x1 * cos_theta
            
            actual_0 = y[0, pos_idx, pair_idx*2].item()
            actual_1 = y[0, pos_idx, pair_idx*2 + 1].item()
            
            assert math.isclose(expected_0, actual_0, rel_tol=1e-5), \
                f"Mismatch at pos {pos}, pair {pair_idx}, dim {pair_idx*2}: expected {expected_0}, got {actual_0}"
            assert math.isclose(expected_1, actual_1, rel_tol=1e-5), \
                f"Mismatch at pos {pos}, pair {pair_idx}, dim {pair_idx*2+1}: expected {expected_1}, got {actual_1}"
    
    print("✅ RoPE implementation correctness test passed!")


if __name__ == '__main__':
    test_simple_rope() 
    test_manual_pi_half_rotation()
    test_different_positions_change_output()
    test_rope_correctness()