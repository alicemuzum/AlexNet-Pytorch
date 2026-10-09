import torch.nn as nn

# input size should be : (b x 3 x 227 x 227)
# The image in the original paper states that width and height are 224 pixels, but
# the dimensions after first convolution layer do not lead to 55 x 55.

class AlexNet(nn.Module):
    def __init__(self, num_classes,input_ch=3,):
        super(AlexNet,self).__init__()
        self.input_ch = input_ch
        self._create_net()
        
    def forward(self,x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.linear(out)
        return out 
    
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
            nn.Dropout(0.8),
            nn.Linear(4096,2048),
            nn.ReLU(),
            nn.Linear(2048,20)
        )

  
        
