import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
# class laplacian(nn.Module):
#     def __init__(self, channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
#         super(laplacian, self).__init__()
#         kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
#         self.convl = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
#         self.convl.weight.data.copy_(torch.from_numpy(kernel))
#     def forward(self, x):
#         x = self.convl(x)
#         x = torch.abs(x)
#         return x

# if __name__ == '__main__':
#     x = torch.randn([4, 16, 480, 640])
#     # model = laplacian(x.size()[1])
#     # print(model(x))

x = torch.tensor(np.random.rand(3,96,480,640).astype(np.float32))
x1 = torch.max(x, 1, True)
x2 = torch.max(x, 1, True)[0]
print(x1)
print('----------------------------------------------------')
print(x2)