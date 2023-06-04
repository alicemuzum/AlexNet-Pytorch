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
        self.conv = self._create_conv_net(self.config)
        self.linear = self._create_linear_net(num_classes)
        self.init_bias()
        
    def forward(self,x):
        return self.linear(self.conv(x))
    
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

    def _create_conv_net(self, config):

        layers = []
        input_ch = self.input_ch

        for idx, block in enumerate(config):
            if type(block) == tuple:        
                layers += [
                    nn.Conv2d(input_ch, block[1],kernel_size=block[0],
                               stride=block[2],padding=block[3]),
                ]
                layers += [nn.ReLU()]
                input_ch = block[1]

            elif type(block) == str:
                if block == "M":
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2)]

                if block == "N":
                    layers += [nn.LocalResponseNorm(k=2,size=5)]
        
        return nn.Sequential(*layers)

    def _create_linear_net(self, num_classes):
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(6*6*256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
                  
