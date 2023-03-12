import math
import torch
def linear_beta_schedule(timesteps):
    """
    linear schedule, proposed in original ddpm paper
    [0.0001, 0.00012, 0.00014, ..., 0.0199]
    """
    scale = 1000 / timesteps # 1.0
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype = torch.float64)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    Improved Denoising Diffusion Probabilistic
    [] 
    """
    steps = timesteps + 1
    #[0, 0.001, 0.002, ..., 1]
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    #[0.008/1.008, 0.009/1.008, 0.010/1.008, ..., 1.008/1.008] * 0.5 pi
    # 
    #[0.999, 0.998, ..., 0.01, 0.0], 数值前大后小，变化前慢后快
    #[0.998, 0.997, ..., 0.0001, 0.0]， 加速衰减
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    #[1.0, 0.999, ..., 0.0009, 0.0004, 0.0001, 0.0]
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    #[0.001, 0.002, ..., 0.55, 0.75, 1.0] 
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def sigmoid_beta_schedule(timesteps, start = -3, end = 3, tau = 1, clamp_min = 1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    Scalable Adaptive Computation for Iterative Generation
    RIN - Recurrent Interface Network
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype = torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

import numpy
import matplotlib.pyplot as plt

#plt.figure(i+1)
timesteps = 100
for i,f in enumerate([linear_beta_schedule, cosine_beta_schedule, sigmoid_beta_schedule]):
    print(f(timesteps))
    plt.subplot(410+i+1)
    plt.plot(f(timesteps))
    #print(f(timesteps).gather(-1, torch.Tensor([0,1,2,4,8,16,32,128,256,512,999], dtype=torch.int64)))
d = 10**(3.0 / timesteps**2)
s = torch.Tensor([d**(i**2-timesteps**2) for i in range(timesteps)])
print(s)
plt.subplot(414)
plt.plot(s)
plt.show()
