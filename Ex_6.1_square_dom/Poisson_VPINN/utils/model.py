import torch
import numpy as np

    

#Define NN models. Maps from R2 -> R1.
class NNsol_hmg(torch.nn.Module):
    def __init__(self):
        super(NNsol_hmg,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,1)

    def forward(self,x):
        x1 = torch.tanh(self.L1(x))

        x2 = self.L2(x1)

        x3 = x2*x[:,:,0].unsqueeze(-1)*(1-x[:,:,0].unsqueeze(-1))*x[:,:,1].unsqueeze(-1)*(1-x[:,:,1].unsqueeze(-1))

        return x3

class NNsol(torch.nn.Module):
    def __init__(self):
        super(NNsol,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,1)

    def forward(self,x,bdrynet):
        x1 = torch.tanh(self.L1(x))

        x2 = self.L2(x1)

        x3 = x2*x[:,:,0].unsqueeze(-1)*(1-x[:,:,0].unsqueeze(-1))*x[:,:,1].unsqueeze(-1)*(1-x[:,:,1].unsqueeze(-1))

        
        extension = bdrynet(x)
        return x3 + extension

class NNbdry(torch.nn.Module):
    def __init__(self):
        super(NNbdry,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,1)

    def forward(self,x):
        x1 = torch.tanh(self.L1(x))

        x2 = self.L2(x1)

        return x2
    

def projection_softmax(net_values,low,high,a,hard_constraint=False):
    m = (high+low)/2.
    delta = high - m
    if not hard_constraint:
        b = 2*delta*(1+np.exp(-a*delta))/(1-np.exp(-a*delta))
        sig_x = torch.sigmoid(a*(net_values-m)) - 0.5
        out = b*sig_x + m
    else:
        b = 2*(high-m)
        sig_x = torch.sigmoid(a*(net_values-m)) - 0.5
        out = b*sig_x + m

    return out

def projection_clamp(net_values,low,high):
    out = torch.clamp(net_values,low,high)

    return out


def init_weights(m):
    if isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
    