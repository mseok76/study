import torch
import torch.nn as nn


class Sobel_filter(nn.Module):
    def __init__(self):
        super(Sobel_filter, self).__init__()

        self.convnet1 = nn.Conv2d(1,1, kernel_size = 3, padding = 1, bias = False)  #width
        self.convnet2 = nn.Conv2d(1,1, kernel_size = 3, padding = 1, bias = False)  #height
        sobel_kernel1 = torch.tensor([[-1.,0.,1.],
                                           [-2.,0.,2.],
                                           [-1.,0.,1.]]).view(1,1,3,3) ##output,input,height,width
        sobel_kernel2 = torch.tensor([[1.,2.,1.],
                                           [0.,0.,0.],
                                           [-1.,-2.,-1.]]).view(1,1,3,3) ##output,input,height,width
        self.convnet1.weight = nn.Parameter(sobel_kernel1)
        self.convnet2.weight = nn.Parameter(sobel_kernel2)

    def forward(self, input):
        edge_x = self.convnet1(input)
        edge_y = self.convnet2(input)
        edge = (edge_x**2 + edge_y**2)**(1/2)

        return edge

