import numpy as np
from math import cos,sin 

def make_toy_data(T=100,change_point = 40):
    z_init = np.array([[5.0],[0.0]])
    alpha1 = 0.1
    rotation = np.array([[cos(alpha1),-sin(alpha1)],
        [sin(alpha1),cos(alpha1)]])
    lambda_before = 1.0
    lambda_after = 0.99
    A_before = lambda_before * rotation 
    A_after = lambda_after * rotation
    b_before = np.array([[0.0],[0.0]])
    b_after = np.array([[0.0],[0.0]])
    B= np.array([[0.7,0.3],[0.8,0.2],[0.2,0.8]])
   
    def transition(z,A,b,eps):
        return A @ (z - b) + b + np.random.randn(2,1) * eps

    def emmition(B,z):
        return B @ z + np.random.randn(3,1)
    
    eps = 0.05
    _z_seq=[z_init]
    _x_seq=[]
    for t in range(T):
        _x_seq.append(emmition(B,_z_seq[-1]))
        if t < change_point:
            _z_seq.append(transition(_z_seq[-1],A_before,b_before,eps))
        else:
            _z_seq.append(transition(_z_seq[-1],A_after,b_after,eps))
    z_seq = np.array(_z_seq)
    z_seq = x_seq.reshape(T,1,3)
    x_seq = np.array(_x_seq)
    x_seq = x_seq.reshape(T,1,3)
    np.save("z_seq_"+str(T)+"_"+str(change_point)+".npy",z_seq)
    np.save("x_seq_"+str(T)+"_"+str(change_point)+".npy",x_seq)
