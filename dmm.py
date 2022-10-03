import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.optim import ClippedAdam
from pyro.infer import SVI,Trace_ELBO
from tqdm import tqdm

class Emittion(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_z_to_x = nn.Linear(2,3)
    def forward(self, z):
        mean = self.lin_z_to_x(z)
        return mean

class Transition(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin_z_to_z = nn.Linear(2,2)
    def forward(self, z):
        mean = self.lin_z_to_z(z)
        return mean

class DMM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emittion = Emittion()
        self.transition = Transition()
        self.length = 1000
        self.rnn = nn.RNN(
            input_size=3,
            hidden_size=2,
            batch_first=False,
            bidirectional=False
        )
        self.z_0 = nn.Parameter(torch.zeros(2))
        self.h_0 = nn.Parameter(torch.zeros((1, 1, 2)))

    def model(self,data):        
        pyro.module("dmm", self)
        z_prev = self.z_0
        for t in range(1, self.length):
            z_loc = self.transition(z_prev)
            z_t = pyro.sample("Z_%d" % t,dist.Normal(z_loc,0.01).to_event(1))           
            emission_probs_t = self.emittion(z_t)
            x_t = pyro.sample(
                "X_%d" % t,
                dist.Normal(emission_probs_t,0.01).to_event(1),
                obs=data[t]
            )
            z_prev = z_t

    def guide(self,data):
        pyro.module("dmm", self)
        h0 = self.h_0
        rnn_output, _ = self.rnn(data,h0)      
        for t in range(1, self.length):
            z_loc = rnn_output[t-1]
            pyro.sample("Z_%d" % t,dist.Normal(z_loc,0.01).to_event(1))

def main():
    data = np.load("toy_data/x_seq.npy")
    data = data.reshape(1000,1,3)
    data = torch.from_numpy(data).float()
    dmm = DMM()
    adam_params = {
        "lr": 0.003,
        "clip_norm": 1.0
    }
    adam = ClippedAdam(adam_params)
    elbo = Trace_ELBO()
    svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)
    with open("dmm.log","w") as f:
        for _ in tqdm(range(10000)):
            loss = svi.step(data)
            print(loss,file=f)
    
if __name__ == "__main__":
    main()

