#backpropagation
#now the loss keep decreasing
#can we smooth out the 
import BaseResNet50
import RNN_softmax
from clip import clip
import torch
from torch import nn
from PIL import Image
import torch.optim as optim

device = "cuda" if torch.cuda.is_available() else "cpu"

class Mirror(nn.Module):
    def __init__(self,name="mirror"):
        super().__init__()
        self.cnn=BaseResNet50.BaseNet()
        self.img_proj=nn.Linear(2048,512)
        self.rnn_softmax=RNN_softmax.RNN_softmax(input_size=512,hidden_size=256,proj_size=10)
        self.clip_,self.preprocess = clip.load("RN50", device=device)
        for param in self.cnn.parameters():
            param.requires_grad = False
        for param in self.clip_.parameters():
            param.requires_grad = False

    def forward(self,x):# x: raw img
        x1=self.preprocess(x).unsqueeze(0).to(device)
        x2=self.cnn(x1)
        x3=self.img_proj(x2)
        x4=self.rnn_softmax(x3)
        #txt_=self.clip_.encode_text_year(x4)
        txt_=self.clip_.encode_text_year_window(x4,window=5)
        txt_=torch.cat([self.clip_.encode_text_year(x4),txt_],dim=0)
        print("out",txt_.shape)
        img_=self.clip_.encode_image(x1)
        return txt_,img_
        

steps=10000

def main():
    img=Image.open("./img/vincent-van-gogh_self-portrait-with-straw-hat-1887.jpg")
    mirror_net = Mirror()
    optimizer = optim.SGD(mirror_net.parameters(), lr=0.05)
    for stps in range(steps):
        optimizer.zero_grad()
        txt_,img_=mirror_net(img)
        cosine_similarity_module = nn.CosineSimilarity(dim=1)
        sim_loss=-1000*torch.mean(cosine_similarity_module(txt_,img_))
        print("similarity loss",sim_loss)
        sim_loss.backward()#backpropagation compute
#        for param in mirror_net.rnn_softmax.rnn.parameters():#why not trained?
#            print("grad",param.grad)
        optimizer.step()#update
        

if __name__ == "__main__":
    main()
