from tokenize import Double
import numpy as np
import torch
import torch.nn as nn
import pyro
import pyro.distributions as dist
from pyro.optim import ClippedAdam
from pyro.infer import SVI,Trace_ELBO

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
        print(type(self.rnn.input_size))
        print(type(self.rnn.dropout))
        
                
        self.z_0 = nn.Parameter(torch.zeros(2))
        self.z_q_0 = nn.Parameter(torch.zeros(2))
        self.h_0 = nn.Parameter(torch.zeros((1, 1, 2),dtype=torch.float64))
        

    def model(self,data):        
        pyro.module("dmm", self)
        z_prev = self.z_0
        for t in pyro.markov(range(1, self.length)):
            z_loc = self.trans(z_prev)
            z_t = pyro.sample(dist.Normal(z_loc))
            emission_probs_t = self.emittion(z_t)
            # the next statement instructs pyro to observe x_t according to the
            # bernoulli distribution p(x_t|z_t)
            pyro.sample(
                "obs_x_%d" % t,
                dist.Normal(emission_probs_t)
            )
            z_prev = z_t

    # q(z_{1:T} | x_{1:T}) (i.e. the variational distribution)
    def guide(self,data):
        pyro.module("dmm", self)
        h0 = self.h_0
        h0 = self.h_0.expand(
            1, 1, self.rnn.hidden_size
        )
        print(data.dtype)
        rnn_output, _ = self.rnn(data,h0)      
        #z_prev = self.z_q_0
        #for t in pyro.markov(range(1, self.length)):
        z_loc = rnn_output[:, self.length - 1, :]
        z_dist = dist.Normal(z_loc)
        pyro.sample("z_%d" % self.length, z_dist())
            

def main():
    data = np.load("toy_data/x_seq.npy")

    data = data.reshape(1000,1,3)

    data = torch.tensor(data)

  
    
    
    dmm = DMM()
    adam_params = {
        "lr": 0.003,
        "clip_norm": 1.0
    }
    adam = ClippedAdam(adam_params)
    elbo = Trace_ELBO()
    svi = SVI(dmm.model, dmm.guide, adam, loss=elbo)
    for _ in range(100):

        loss = svi.step(data)
        print(loss)
    
if __name__ == "__main__":
    main()

