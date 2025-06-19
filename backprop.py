#backpropagation
import BaseResNet50
import RNN_GUMBEL
import clip
import torch
from torch import nn
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

class Mirror(nn.Module):
    def __init__(self,name="mirror"):
        super().__init__()
        self.cnn=BaseResNet50.BaseNet()
        self.img_proj=nn.Linear(2048,512)
        self.rnn_gumbel=RNN_GUMBEL.RNN_gumbel(input_size=512,hidden_size=256,proj_size=10)
        self.clip_,self.preprocess = clip.load("RN50", device=device)
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.clip_.parameters():
            param.requires_grad = False

    def forward(self,x):# x: raw img
        x=self.preprocess(x).unsqueeze(0).to(device)
        x=self.cnn(x)
        x=self.img_proj(x)
        x=self.rnn_gumbel(x)
        return x
        #txt for inner product
        #img for inner product
        

steps=2
def main():
    mirror_net=Mirror()
    for stps in range(steps):
        txt_=mirror_net(Image.open("./img/vincent-van-gogh_self-portrait-with-straw-hat-1887.jpg"))
        print(txt_)
        print(txt_.shape)
        


    
    

    
    

if __name__ == "__main__":
    main()
