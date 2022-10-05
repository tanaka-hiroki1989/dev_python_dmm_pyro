import numpy as np
import matplotlib.pyplot as plt
 
x = y = [-3,0,3]
 
x, y = np.meshgrid(x, y)
u = -y
v = x

#u, v = np.meshgrid(u, v)
 
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
 
lim = 8
for ax in axes:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
 
    ax.set_xticks(np.arange(-lim, lim, 1))
    ax.set_yticks(np.arange(-lim, lim, 1))
 
    ax.grid()
    ax.set_aspect('equal')
 
C = np.sqrt(u * u + v * v)
axes[0].streamplot(x, y, u, v, density=[0.2, 0.2], broken_streamlines=False)

 
plt.show()