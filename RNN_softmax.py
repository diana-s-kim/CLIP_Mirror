import torch
from torch import nn
from torch.nn import functional as F

class RNN_softmax(nn.Module):
    def __init__(self,name="rnn_softmax",input_size=None,hidden_size=None,proj_size=None):
        super().__init__()
        self.rnn=nn.LSTM(input_size=input_size,hidden_size=hidden_size,proj_size=proj_size)
        self.embedding=nn.Embedding(10,input_size)
        self.softmax=nn.Softmax(dim=1)
        self.h0 = torch.randn(1,proj_size)
        self.c0 = torch.randn(1,hidden_size)
        
    def forward(self,x):
        y1, (h1, c1) =self.rnn(x, (self.h0, self.c0))
        o1=self.softmax(y1)

        i1=o1@self.embedding.weight
        y2, (h2, c2) = self.rnn(i1, (h1, c1))
        o2=self.softmax(y2)

        i2=o2@self.embedding.weight
        y3, (h3, c3)= self.rnn(i2, (h2, c2))
        o3=self.softmax(y3)

        i3=o3@self.embedding.weight
        y4, (h4, c4)= self.rnn(i3, (h3, c3))
        o4=self.softmax(y4)
        
        out=torch.cat((o1,o2,o3,o4),dim=0)
        print("softmax-out",out.argmax(dim=1))#embedding may be the blending (simplex)
        return out
