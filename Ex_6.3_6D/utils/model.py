import torch
import numpy as np

class NN(torch.nn.Module):
    def __init__(self):
        super(NN,self).__init__()
        self.L1 = torch.nn.Linear(6,80)
        self.L2 = torch.nn.Linear(80,80)
        #self.L3 = torch.nn.Linear(80,80)
        self.L4 = torch.nn.Linear(80,80)
        self.L5 = torch.nn.Linear(80,1)

    def forward(self,xi1,xi2,xi3,xi4,xi5,xi6):
        inputs = torch.cat([xi1,xi2,xi3,xi4,xi5,xi6], axis=1)
        x1 = torch.tanh(self.L1(inputs))
        x2 = torch.tanh(self.L2(x1))
        #x3 = torch.tanh(self.L3(x2))
        x4 = torch.tanh(self.L4(x2))
        x5 = self.L5(x4)

        return x5    

class NN_WAN(torch.nn.Module):
    def __init__(self):
        super(NN_WAN,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30) 
        self.L3 = torch.nn.Linear(30,30)
        self.L4 = torch.nn.Linear(30,30)
        self.L5 = torch.nn.Linear(30,1)

    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)
        x1 = torch.tanh(self.L1(inputs))

        x2 = torch.tanh(self.L2(x1)) 
        x2 = torch.tanh(self.L3(x2))

        x3 = torch.tanh(self.L4(x2))
        x3 = self.L5(x3)

        return x3

class NNbdry(torch.nn.Module):
    def __init__(self):
        super(NNbdry,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,1)

    def forward(self,x):
        x1 = torch.tanh(self.L1(x))

        x2 = self.L2(x1)

        return x2

class NN_VPINN(torch.nn.Module):
    def __init__(self):
        super(NN_VPINN,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30)
        self.L3 = torch.nn.Linear(30,1)

    def forward(self,x,bdrynet):
        x1 = torch.tanh(self.L1(x))

        x2 = self.L2(x1)
        x2 = self.L3(x2)
    

        x3 = x2*x[:,:,0].unsqueeze(-1)*(1-x[:,:,0].unsqueeze(-1))*x[:,:,1].unsqueeze(-1)*(1-x[:,:,1].unsqueeze(-1))

        
        extension = bdrynet(x)
        return x3 + extension

class NN_v_forcing(torch.nn.Module):
    def __init__(self):
        super(NN_v_forcing,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30)
        self.L3 = torch.nn.Linear(30,1)

    def forward(self,x,bdrynet,bdry_hom):
        x1 = torch.tanh(self.L1(x))

        x2 = self.L2(x1)
        x2 = self.L3(x2)
    
        
        extension = bdrynet(x)
        hom = bdry_hom(x)
        return x2*hom + extension

class NN_v_penalty(torch.nn.Module):
    def __init__(self):
        super(NN_v_penalty,self).__init__()
        self.L1 = torch.nn.Linear(2,30)
        self.L2 = torch.nn.Linear(30,30)
        #self.L3 = torch.nn.Linear(30,30)
        #self.L4 = torch.nn.Linear(30,30)
        self.L5 = torch.nn.Linear(30,1)

    def forward(self,x):
        x1 = torch.tanh(self.L1(x))
        x2 = torch.tanh(self.L2(x1))
        #x3 = torch.tanh(self.L3(x2))
        #x4 = torch.tanh(self.L4(x3))
        x5 = self.L5(x2)

        return x5    


class adverse(torch.nn.Module):
    def __init__(self):
        super(adverse,self).__init__()
        self.input_layer = torch.nn.Linear(2,30)
        self.Hidden1 = torch.nn.Linear(30,30) 
        self.Hidden2 = torch.nn.Linear(30,30)
        self.Hidden3 = torch.nn.Linear(30,30)
        self.output_layer = torch.nn.Linear(30,1)


    def forward(self,x,y):
        inputs = torch.cat([x,y],axis=1)

        x1 = torch.tanh(self.input_layer(inputs))

        x2 = torch.tanh(self.Hidden1(x1)) + x1
        x2 = torch.tanh(self.Hidden2(x2)) 
        x3 = torch.tanh(self.Hidden3(x2)) + x2

        x3 = self.output_layer(x3)

        out = x3 * torch.sin(torch.pi*x) * torch.sin(torch.pi*y)

        return out

def init_weights(m):
    if isinstance(m,torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)
