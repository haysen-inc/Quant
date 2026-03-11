import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class DifferentiableEMA(nn.Module):
    """
    Differentiable Exponential Moving Average.
    Instead of a fixed integer period N, it learns a weight w_alpha.
    alpha = sigmoid(w_alpha)
    We initialize w_alpha using Logit to exactly match 2/(N+1) at Epoch 0.
    """
    def __init__(self, init_n):
        super().__init__()
        # Inverse mapping: w = ln(p / (1-p)) where p = 2/(N+1)
        p = 2.0 / (init_n + 1.0)
        init_w = math.log(p / (1.0 - p))
        self.w_alpha = nn.Parameter(torch.tensor(init_w, dtype=torch.float32))

    def forward(self, tensor):
        alpha = torch.sigmoid(self.w_alpha)
        res = torch.empty_like(tensor)
        if len(tensor) == 0:
            return res
        
        valid_mask = ~torch.isnan(tensor)
        if not valid_mask.any():
            return res.fill_(float('nan'))
        
        first_valid_idx = valid_mask.nonzero()[0].item()
        res[:first_valid_idx] = float('nan')
        res[first_valid_idx] = tensor[first_valid_idx]
        
        val = tensor[first_valid_idx].item()
        for i in range(first_valid_idx + 1, len(tensor)):
            x = tensor[i].item()
            if torch.isnan(torch.tensor(x)):
                res[i] = val
            else:
                val = alpha * x + (1 - alpha) * val
                res[i] = val
        return res

class SoftAttentionHHV(nn.Module):
    """
    Differentiable Highest High Value using Soft-Attention.
    Temperature tau determines how sharply to pick the absolute max.
    As tau -> 0, it behaves exactly like max(X).
    """
    def __init__(self, n, init_tau=-10.0):
        super().__init__()
        self.n = n
        # Softplus(w_tau) guarantees tau is strictly > 0.
        # Initialize with -10.0 makes tau very close to 0, mimicking exact max.
        self.w_tau = nn.Parameter(torch.tensor(init_tau, dtype=torch.float32))

    def forward(self, tensor):
        tau = F.softplus(self.w_tau) + 1e-5
        
        padded = F.pad(tensor, (self.n - 1, 0), value=float('-inf'))
        unfolded = padded.unfold(0, self.n, 1) # [L, n]
        
        # Soft-Attention calculation: sum(x * exp(x/tau)) / sum(exp(x/tau))
        weights = F.softmax(unfolded / tau, dim=1)
        soft_max_vals = (unfolded * weights).sum(dim=1)
        return soft_max_vals

class SoftAttentionLLV(nn.Module):
    """
    Differentiable Lowest Low Value using Soft-Attention.
    We invert the signal: softmax(-x / tau) selects the minimum.
    """
    def __init__(self, n, init_tau=-10.0):
        super().__init__()
        self.n = n
        self.w_tau = nn.Parameter(torch.tensor(init_tau, dtype=torch.float32))

    def forward(self, tensor):
        tau = F.softplus(self.w_tau) + 1e-5
        
        padded = F.pad(tensor, (self.n - 1, 0), value=float('inf'))
        unfolded = padded.unfold(0, self.n, 1) # [L, n]
        
        # Soft-Attention calculation for Minimum
        weights = F.softmax(-unfolded / tau, dim=1)
        soft_min_vals = (unfolded * weights).sum(dim=1)
        return soft_min_vals

def REF(tensor, n):
    """
    Returns the value of the tensor shifted backwards by n periods.
    Fills the first n elements with NaN.
    """
    assert len(tensor.shape) == 1
    res = torch.empty_like(tensor).fill_(float('nan'))
    if n < len(tensor):
        res[n:] = tensor[:-n]
    return res

def EMA(tensor, n):
    """
    Exponential Moving Average.
    Alpha = 2 / (n + 1)
    """
    alpha = 2.0 / (n + 1)
    res = torch.empty_like(tensor)
    if len(tensor) == 0:
        return res
    
    # Initialize with the first valid (non-NaN) value or just index 0
    # Handle NaNs by skipping them to find the first valid
    valid_mask = ~torch.isnan(tensor)
    if not valid_mask.any():
        return res.fill_(float('nan'))
    
    first_valid_idx = valid_mask.nonzero()[0].item()
    res[:first_valid_idx] = float('nan')
    res[first_valid_idx] = tensor[first_valid_idx]
    
    # For a purely PyTorch approach, we can iterate
    # Or use a cumsum trick, but a compiled loop for 1D is simplest
    # Converting to Python list or using a simple loop for standard 1D array is fast enough for small/med arrays (<1M elements).
    val = tensor[first_valid_idx].item()
    for i in range(first_valid_idx + 1, len(tensor)):
        x = tensor[i].item()
        if torch.isnan(torch.tensor(x)):
            res[i] = val
        else:
            val = alpha * x + (1 - alpha) * val
            res[i] = val
    return res

def SMA(tensor, n, m):
    """
    A specific smoothed moving average often used in MyLanguage/Pine.
    Y = (M * X + (N - M) * Y') / N
    Alpha = m / n
    """
    alpha = m / n
    res = torch.empty_like(tensor)
    
    valid_mask = ~torch.isnan(tensor)
    if not valid_mask.any():
        return res.fill_(float('nan'))
    
    first_valid_idx = valid_mask.nonzero()[0].item()
    res[:first_valid_idx] = float('nan')
    res[first_valid_idx] = tensor[first_valid_idx]
    
    val = tensor[first_valid_idx].item()
    for i in range(first_valid_idx + 1, len(tensor)):
        x = tensor[i].item()
        if torch.isnan(torch.tensor(x)):
            res[i] = val
        else:
            val = alpha * x + (1 - alpha) * val
            res[i] = val
    return res

def HHV(tensor, n):
    """
    Highest value over the last n periods.
    """
    # unfold shape will be [L - n + 1, n]
    # we need padding at the beginning to maintain length
    padded = F.pad(tensor, (n - 1, 0), value=float('-inf'))
    unfolded = padded.unfold(0, n, 1)
    # Get max along the window dimension
    max_vals, _ = unfolded.max(dim=1)
    return max_vals

def LLV(tensor, n):
    """
    Lowest value over the last n periods.
    """
    padded = F.pad(tensor, (n - 1, 0), value=float('inf'))
    unfolded = padded.unfold(0, n, 1)
    min_vals, _ = unfolded.min(dim=1)
    return min_vals

def STDP(tensor, n):
    """
    Population standard deviation over n periods.
    """
    padded = F.pad(tensor, (n - 1, 0), value=float('nan'))
    unfolded = padded.unfold(0, n, 1)
    # std(unbiased=False) gives population std
    return unfolded.std(dim=1, unbiased=False)

def MA_DYNAMIC(tensor, window_sizes):
    """
    Moving average where the window size varies at each time step.
    window_sizes is a tensor of the same length as `tensor`.
    """
    res = torch.empty_like(tensor).fill_(float('nan'))
    for i in range(len(tensor)):
        if not torch.isnan(window_sizes[i]):
            w = int(window_sizes[i].item())
            if w > 0:
                start_idx = max(0, i - w + 1)
                res[i] = tensor[start_idx:i+1].mean()
            else:
                res[i] = tensor[i]
    return res

def MA(tensor, n):
    """
    Simple moving average over fixed window.
    """
    padded = F.pad(tensor, (n - 1, 0), value=float('nan'))
    unfolded = padded.unfold(0, n, 1)
    return unfolded.mean(dim=1)

def CROSS(a, b):
    """
    Returns 1.0 when a crosses above b, else 0.0.
    """
    a_prev = REF(a, 1)
    b_prev = REF(b, 1)
    cross_mask = (a_prev <= b_prev) & (a > b)
    return cross_mask.float()

def CROSSDOWN(a, b):
    """
    Returns 1.0 when a crosses below b, else 0.0.
    """
    a_prev = REF(a, 1)
    b_prev = REF(b, 1)
    cross_mask = (a_prev >= b_prev) & (a < b)
    return cross_mask.float()

def EVERY(cond, n):
    """
    Returns 1.0 if cond is true for the last n periods continuously.
    cond is a boolean or float tensor (1.0 = true, 0.0 = false)
    """
    cond_float = cond.float()
    padded = F.pad(cond_float, (n - 1, 0), value=0.0)
    unfolded = padded.unfold(0, n, 1)
    # every is true if sum == n
    return (unfolded.sum(dim=1) == n).float()

def BARSLAST(cond):
    """
    Returns the number of periods since the condition last triggered.
    If condition has never triggered, returns NaN or a large number.
    Here we return NaN before the first trigger, and 0 on the bar it triggers.
    """
    res = torch.empty_like(cond, dtype=torch.float32).fill_(float('nan'))
    last_trigger = -1
    for i in range(len(cond)):
        if cond[i] > 0:
            last_trigger = i
        
        if last_trigger != -1:
            res[i] = i - last_trigger
    return res

def MIN(a, b):
    return torch.min(a, b)

def MAX(a, b):
    return torch.max(a, b)
