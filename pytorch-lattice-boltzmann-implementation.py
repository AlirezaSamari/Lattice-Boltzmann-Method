import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

maxIter = 50000  # total number of time iteration
nx, ny = 100, 100  # number of lattice nodes
cs = np.sqrt(1/3)
nulb = 0.01  # viscosity in lattice units
tau = 0.5 + nulb/cs**2
rho0 = 5
u0 = 0.05
Re = (u0 * ny)/nulb
print(f'Re = {Re}')
fin = torch.zeros((9, nx, ny))

# Lattice Constants
v = torch.tensor([[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, -1]], dtype=torch.float32)
t = torch.tensor([1/36, 1/9, 1/36, 1/9, 4/9, 1/9, 1/36, 1/9, 1/36], dtype=torch.float32)

# Function Definitions
def solid(nx, ny):
    solid = torch.zeros((nx, ny), dtype=torch.bool)
    solid[:, 0] = True
    solid[-1, :] = True
    solid[0, :] = True
    return solid

def macroscopic(fin):
    rho = fin.sum(dim=0)
    u = torch.zeros((2, nx, ny))
    for i in range(9):
        u[0, :, :] += v[i, 0] * fin[i, :, :]
        u[1, :, :] += v[i, 1] * fin[i, :, :]
    u /= rho
    return rho, u

def equilibrium(rho, u):
    usqr = (3/2) * (u[0]**2 + u[1]**2)
    feq = torch.zeros((9, nx, ny))
    for i in range(9):
        uv = 3 * (v[i, 0] * u[0, :, :] + v[i, 1] * u[1, :, :])
        feq[i, :, :] = rho * t[i] * (1 + uv + 0.5 * uv**2 - usqr)
    return feq

# Setup
solid_mask = solid(nx, ny)
for i in range(9):
    fin[i, :, :] = t[i] * rho0
rho = torch.ones((nx, ny)) * rho0
u = torch.ones((2, nx, ny)) * 0

# Main Time Loop
x = torch.arange(0, nx, 1)
y = torch.arange(0, ny, 1)
X, Y = torch.meshgrid(x, y)
for time in range(maxIter + 1):
    rho, u = macroscopic(fin)
    u[0, :, -1] = u0

    # compute equilibrium
    feq = equilibrium(rho, u)

    # collision step
    fout = fin - (fin - feq)/tau

    for i in range(9):
        # Streaming step
        fin[i, :, :] = torch.roll(torch.roll(fout[i, :, :], int(v[i, 0].item()), dims=0), int(v[i, 1].item()), dims=1)
        # bounce-back
        fin[i][solid_mask] = fout[8 - i][solid_mask]
        # Moving-wall boundary condition
        fin[i, :, -1] = fout[8 - i, :, -1] - 6 * rho[:, -1] * t[8 - i] * v[8 - i, 0] * u0

    if time % 10000 == 0:
        print(f'Iteration = {time}')

x = np.arange(0, nx, 1)
y = np.arange(0, ny, 1)
X, Y = np.meshgrid(x, y)

fig, axs = plt.subplots(2, 3, figsize=(12, 8))

axs[0, 0].set_title('Velocity Field')
axs[0, 0].set_xlabel('X')
axs[0, 0].set_ylabel('Y')
imgplot1 = axs[0, 0].imshow(torch.sqrt(u[0]**2 + u[1]**2).cpu().T.numpy(), cmap='jet')
fig.colorbar(imgplot1, ax=axs[0, 0])

axs[0, 1].set_title('Horizontal Velocity')
axs[0, 1].set_xlabel('X')
axs[0, 1].set_ylabel('Y')
imgplot2 = axs[0, 1].imshow((u[0]).cpu().T.numpy(), cmap='jet')
fig.colorbar(imgplot2, ax=axs[0, 1])

axs[0, 2].set_title('Density Field')
axs[0, 2].set_xlabel('X')
axs[0, 2].set_ylabel('Y')
imgplot3 = axs[0, 2].imshow(rho.cpu().T.numpy(), cmap='jet')
fig.colorbar(imgplot3, ax=axs[0, 2])

axs[1, 0].set_title('Horizontal Velocity in middle of X-axis')
axs[1, 0].set_xlabel('Y')
axs[1, 0].set_ylabel('Horizontal Velocity')
axs[1, 0].plot(u[0, 50, :].cpu(), list(range(0, 100)), color='red')

axs[1, 1].set_title('Vertical Velocity in middle of Y-axis')
axs[1, 1].set_xlabel('X')
axs[1, 1].set_ylabel('Vertical Velocity')
axs[1, 1].plot(u[1, :, 50].cpu(), list(range(0, 100)))

axs[1, 2].set_title('Streamlines')
axs[1, 2].set_xlabel('X')
axs[1, 2].set_ylabel('Y')
axs[1, 2].streamplot(X, Y, u[0, :, :].cpu().T.numpy(), u[1, :, :].cpu().T.numpy(), density=1.1, arrowstyle='-', linewidth=0.9)

plt.tight_layout()
plt.show()