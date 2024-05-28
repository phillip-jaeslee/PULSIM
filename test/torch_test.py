import torch
import numpy as np
import matplotlib.pyplot as plt
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Rx(theta):
    return torch.stack([
        torch.stack([torch.ones_like(theta), torch.zeros_like(theta), torch.zeros_like(theta)], dim=0),
        torch.stack([torch.zeros_like(theta), torch.cos(theta), torch.sin(theta)], dim=0),
        torch.stack([torch.zeros_like(theta), -torch.sin(theta), torch.cos(theta)], dim=0)
    ], dim=0).to(device)

def Ry(theta):
    return torch.stack([
        torch.stack([torch.cos(theta), torch.zeros_like(theta), torch.sin(theta)], dim=0),
        torch.stack([torch.zeros_like(theta), torch.ones_like(theta), torch.zeros_like(theta)], dim=0),
        torch.stack([-torch.sin(theta), torch.zeros_like(theta), torch.cos(theta)], dim=0)
    ], dim=0).to(device)

def Rz(theta):
    return torch.stack([
        torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=0),
        torch.stack([-torch.sin(theta), torch.cos(theta), torch.zeros_like(theta)], dim=0),
        torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=0)
    ], dim=0).to(device)

def bloch_rotate(M_init, T, B, angle):
    Gamma = 42.58  # kHz/mT MHz/T
    flip = 2 * torch.pi * Gamma * torch.norm(B, dim=1) * T
    eta = torch.acos(B[:, 2] / (torch.norm(B, dim=1) + torch.finfo(torch.float32).eps))
    theta = torch.atan2(B[:, 1], B[:, 0])

    if angle == "x":
        R = torch.bmm(Rz(-theta).permute(2, 0, 1), Ry(-eta).permute(2, 0, 1))
        R = torch.bmm(R, Rz(flip).permute(2, 0, 1))
        R = torch.bmm(R, Ry(eta).permute(2, 0, 1))
        R = torch.bmm(R, Rz(theta).permute(2, 0, 1))
    elif angle == "y":
        R = torch.bmm(Rx(-theta).permute(2, 0, 1), Rz(-eta).permute(2, 0, 1))
        R = torch.bmm(R, Rx(flip).permute(2, 0, 1))
        R = torch.bmm(R, Rz(eta).permute(2, 0, 1))
        R = torch.bmm(R, Rx(theta).permute(2, 0, 1))
    elif angle == "z":
        R = torch.bmm(Ry(-theta).permute(2, 0, 1), Rx(-eta).permute(2, 0, 1))
        R = torch.bmm(R, Ry(flip).permute(2, 0, 1))
        R = torch.bmm(R, Rx(eta).permute(2, 0, 1))
        R = torch.bmm(R, Ry(theta).permute(2, 0, 1))
    else:
        raise ValueError(f'Failed to run the proper Bloch rotation with "{angle}". Please choose among x, y, z coordinates.')

    # Convert M_init to the appropriate data type before the bmm operation
    M_init = M_init.to(torch.float32)

    return torch.bmm(R, M_init.unsqueeze(2)).squeeze(2)


def hard_pulse(M, flip, angle, t_max, N, BW, Gamma):
    start = time.time()

    # Convert inputs to torch tensors and move to device
    M = M.clone().detach().to(device)
    dt = t_max / N
    init = -N / 2
    final = N / 2
    t = torch.arange(init, final, device=device) * dt
    RF = torch.ones((1, int(N)), dtype=torch.float32, device=device)
    RF = (flip * RF) / torch.sum(RF) / (2 * torch.pi * Gamma * dt)
    df = torch.linspace(-BW / 2, BW / 2, steps=1000, device=device)

    # Expand dimensions to align properly for stacking
    RF_expanded = RF.expand(len(df), -1)
    zeros_expanded = torch.zeros(len(df), N, device=device)
    df_expanded = df[:, None].expand(-1, N) / Gamma

    B = torch.stack([RF_expanded, zeros_expanded, df_expanded], dim=2)

    for n in range(len(t)):
        M = bloch_rotate(M.T, dt, B[:, n, :], angle).T

    end = time.time()
    print('elapsed time: {} sec'.format(end - start))

    return M.cpu().numpy(), df.cpu().numpy(), RF.cpu().numpy(), t_max, N


Gamma = 42.58 # kHz/mT
M0 = 1
M_equilibrium = torch.tensor([0, 0, M0], device=device)
BW = 6 # kHz
df = torch.linspace(-BW/2, BW/2, 1000, device=device)
N = 1000

M = torch.tile(M_equilibrium, (len(df), 1)).T

M, df, RF_t, t_max, N = hard_pulse(M, torch.pi, "x", 2, N, BW, Gamma)

t = np.arange(0, N, 1) * t_max / N

fig, axs = plt.subplots(2, 1)
axs[0].plot(t[0]-np.finfo(np.float64).eps, 0)
axs[0].plot(t, RF_t.T)
axs[0].plot(t[-1]+np.finfo(np.float64).eps, 0)
axs[0].set(xlabel='time (ms)', ylabel='RF (mT)')
df = df * 1000
axs[1].plot(df, M[2,:], label="Mz")
axs[1].plot(df, M[1,:], label="My")
axs[1].plot(df, M[0,:], label="Mx")
axs[1].set(xlabel='frequency (Hz)', ylabel='flip')
axs[1].legend()
plt.show()
