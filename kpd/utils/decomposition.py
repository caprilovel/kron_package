import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def fasterKDP(x, a, b):
    """faster calculation of Kronecker product, assume that W = A ⊗ B, and x is the input, then this function will reshape
    x , A and B, make the calculation of W @ x more efficient.

    Args:
        x (torch.tensor): input tensor, shape is [batch_size, ..., a_shape[0] * b_shape[0]]
        a (torch.tensor): factor a, shape is [a_shape[0], a_shape[1]]
        b (torch.tensor): factor b, shape is [b_shape[0], b_shape[1]]

    Returns:
        output (torch.tensor): output tensor, shape is [batch_size, ..., a_shape[1] * b_shape[1]]
    """
    x_shape = x.shape
    a_shape = a.shape
    b_shape = b.shape
    
    
    assert a_shape[0] * b_shape[0] == x_shape[-1], "The shapes of the input tensor and the factors are not compatible"
    # change x[-1] into a[0] b[0]
    x = x.view(-1, a_shape[0], b_shape[0])    
    x = x @ b
    x = torch.permute(x, (0, 2, 1)).contiguous()
    x = x @ a
    x = torch.permute(x, (0, 2, 1)).contiguous()
    
    y_shape = [*x_shape[:-1], a_shape[-1] * b_shape[-1]]
    x = x.view(y_shape)
    return x


def low_rank_approximation(weight, rank):
    """Low rank approximation of the weight matrix, using SVD decomposition.

    Args:
        weight (torch.tensor): weight matrix, shape is [m, n]
        rank (int): rank of the approximation

    Returns:
        u (torch.tensor): factor u, shape is [m, rank]
        v (torch.tensor): factor v, shape is [rank, n]
    """
    U, S, V = torch.linalg.svd(weight)
    u = U[:, :rank]
    v = V[:rank, :]
    s = torch.sqrt(S[:rank])
    u = u @ torch.diag(s)
    v = torch.diag(s) @ v
    return u, v

# def kronecker_approximation()

def factorize(n: int, bias=0) -> List[int]:
    # """Return the most average two factorization of n."""
    for i in range(int(np.sqrt(n)) + 1, 1, -1):
        if n % i == 0:
            if bias == 0:
                return [i, n // i]
            else:
                bias -= 1
    return [n, 1]
