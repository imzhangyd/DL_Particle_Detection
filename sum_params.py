from thop import clever_format
from thop import profile
import torch
from model.superPoint import SuperPointNet

model = SuperPointNet()

input = torch.randn(1, 1, 512, 512)
flops, params = profile(model, inputs=(input, ))
flops, params = clever_format([flops, params], "%.3f")
print(flops)
print(params)