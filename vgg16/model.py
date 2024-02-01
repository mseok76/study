import torch.nn as nn
import torch


def conv2_block(in_dim, out_dim,activation):
    model = nn.Sequential(
        nn.Conv2d(in_dim, out_dim, size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        activation,
        nn.Conv2d(out_dim, out_dim, size = 3, padding = 1),
        nn.BatchNorm2d(out_dim),
        activation,
        nn.MaxPool2d(2,2)
    )
    return model

class vgg_net():
    def __init__(self, act_func= nn.ReLU()):
        super(vgg_net, self).__init__()

        if(act_func == 'ReLU'):
            self.activation = nn.ReLU()
        elif(act_func == 'SiLU'):
            self.activation = nn.SiLU()
        elif(act_func == 'ELU'):
            self.activation = nn.ELU()
        elif(act_func == 'Tanh'):
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError(act_func)
        
        self.convlayers = nn.Sequential(
            conv2_block(3,64),
            conv2_block(64,128),
            conv2_block(128,256),
            conv2_block(256,512),
        )
        self.fclayers = nn.Sequential(
            nn.LazyLinear(120),
            self.activation,
            nn.Linear(120,10)
        )
    
    def forward(self, x):
        x = self.convlayers(x)
        x = x.view(x.size(0),-1)
        x = self.fclayers(x)

        return x
    