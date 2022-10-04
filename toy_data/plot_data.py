import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

def plot_data(T, change_point):
    z_seq = np.load("z_seq_"+str(T)+"_"+str(change_point)+".npy")
    x_seq = np.load("x_seq_"+str(T)+"_"+str(change_point)+".npy")
    z_seq_reshape = z_seq.T
    x_seq_reshape = x_seq.T
    colors = ["blue" if t< change_point else "red" for t in range(T+1)]
    fig = plt.figure() 
    ax = plt.axes()
    ax.grid(color='b', linestyle=':', linewidth=0.3)
    cir1 = patches.Circle(xy=(0.0, 0.0), radius=1.0, fill = False, )
    cir2 = patches.Circle(xy=(-3.0, 0.0), radius=2.0, fill = False)
    ax.scatter(z_seq_reshape[0],z_seq_reshape[1],color=colors,s=0.5)
    ax.set_xlim(-7.5,7.5)
    ax.set_ylim(-7.5,7.5)
    ax.set_aspect("equal")
    plt.savefig("plot/latent_"+str(T)+"_"+str(change_point)+".png")
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x_seq_reshape[0],
        x_seq_reshape[1],
        x_seq_reshape[2],color=colors[:-1],s=1.0)
    plt.savefig("plot/observe_"+str(T)+"_"+str(change_point)+".png")

if __name__ =="__main__":
    plot_data(100,40)
    plot_data(1000,400)