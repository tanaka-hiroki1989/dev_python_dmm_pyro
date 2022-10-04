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

class change_point_DMM(nn.Module):
    def __init__(self,length=1000):
        super().__init__()
        self.emittion = Emittion()
        self.transition_before = Transition()
        self.transition_after = Transition()
        self.length = length
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
        pi = torch.tensor([1.0]*self.length)
        cp = pyro.sample("C",dist.Categorical(torch.softmax(pi,0,dtype=torch.double)))
        z_prev = self.z_0
        for t in range(1, cp+1):
            z_loc = self.transition_before(z_prev)
            z_t = pyro.sample("Z_%d" % t,dist.Normal(z_loc,0.01).to_event(1))           
            emission_probs_t = self.emittion(z_t)
            pyro.sample(
                "X_%d" % t,
                dist.Normal(emission_probs_t,0.01).to_event(1),
                obs=data[t]
            )
            z_prev = z_t
        for t in range(cp+1, self.length):
            z_loc = self.transition_after(z_prev)
            z_t = pyro.sample("Z_%d" % t,dist.Normal(z_loc,0.01).to_event(1))           
            emission_probs_t = self.emittion(z_t)
            pyro.sample(
                "X_%d" % t,
                dist.Normal(emission_probs_t,0.01).to_event(1),
                obs=data[t]
            )
            z_prev = z_t

    def model_long_span(self,data,term=5):        
        pyro.module("dmm", self)
        pi = torch.tensor([1.0]*term)
        cp = pyro.sample("C",dist.Categorical(torch.softmax(pi,0,dtype=torch.double)))
        z_prev = self.z_0
        for t in range(1, 200*cp+1):
            z_loc = self.transition_before(z_prev)
            z_t = pyro.sample("Z_%d" % t,dist.Normal(z_loc,0.01).to_event(1))           
            emission_probs_t = self.emittion(z_t)
            pyro.sample(
                "X_%d" % t,
                dist.Normal(emission_probs_t,0.01).to_event(1),
                obs=data[t]
            )
            z_prev = z_t
        for t in range(200*cp+1, self.length):
            z_loc = self.transition_after(z_prev)
            z_t = pyro.sample("Z_%d" % t,dist.Normal(z_loc,0.01).to_event(1))           
            emission_probs_t = self.emittion(z_t)
            pyro.sample(
                "X_%d" % t,
                dist.Normal(emission_probs_t,0.01).to_event(1),
                obs=data[t]
            )
            z_prev = z_t

    def guide(self,data):
        pyro.module("dmm", self)
        pi = pyro.param('pi',lambda: torch.tensor([1.0]*self.length))
        pyro.sample("C", dist.Categorical(torch.softmax(pi,0,torch.double)))
        h0 = self.h_0
        rnn_output, _ = self.rnn(data,h0)      
        for t in range(1, self.length):
            z_loc = rnn_output[t-1]
            pyro.sample("Z_%d" % t,dist.Normal(z_loc,0.01).to_event(1))
    def long_span_guide(self,data,term=5):
        pyro.module("dmm", self)
        pi = pyro.param('pi',lambda: torch.tensor([1.0]*term))
        pyro.sample("C", dist.Categorical(torch.softmax(pi,0,torch.double)))
        h0 = self.h_0
        rnn_output, _ = self.rnn(data,h0)      
        for t in range(1, self.length):
            z_loc = rnn_output[t-1]
            pyro.sample("Z_%d" % t,dist.Normal(z_loc,0.01).to_event(1))

      
if __name__ == "__main__":
    change_point_DMM(100)

