import torch.nn as nn
import torch

config = [
                    # (3,227,227) input size
    (11, 96, 4, 0), # (96, 55, 55)
    "N",
    "M",            # (96, 27, 27)
    (5, 256, 1, 2), # (256, 27, 27)
    "N",
    "M",            # (256, 13, 13)
    (3, 384, 1, 1), # (384, 13, 13)
    (3, 384, 1, 1), # (384, 13, 13)
    (3, 256, 1, 1), # (256, 13, 13)
    "M"             # (256, 6, 6) output size

]
    
# input size should be : (b x 3 x 227 x 227)
# The image in the original paper states that width and height are 224 pixels, but
# the dimensions after first convolution layer do not lead to 55 x 55.

class AlexNet(nn.Module):
    def __init__(self, num_classes,input_ch=3,):
        super(AlexNet,self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.net = self._create_net()
      
        #self.init_bias()
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.linear(out)
        return out 
    
    def init_bias(self):
        for layer in self.conv:
            if isinstance(layer,nn.Conv2d):
                nn.init.normal_(layer.weight,mean=0,std=0.01)
                nn.init.constant_(layer.bias,0)
        for layer in self.linear:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=0.01)
                nn.init.constant_(layer.bias,1)
        nn.init.constant_(self.conv[4].bias, 1)
        nn.init.constant_(self.conv[10].bias, 1)
        nn.init.constant_(self.conv[12].bias, 1)
        print("Data Norm in weight init:",self.conv[0].weight.data.norm())

    def _create_net(self):

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.LocalResponseNorm(5,alpha=1e-4,beta=0.75,k=2),
            nn.MaxPool2d(3,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96,256,5,padding=2),
            nn.ReLU(),
            nn.LocalResponseNorm(5,alpha=1e-4,beta=0.75,k=2),
            nn.MaxPool2d(3,2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256,384,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,384,3,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,padding=1),
            nn.MaxPool2d(3,2),
        )
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256*6*6,4096),
            nn.ReLU(),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,20)
        )

  
        
