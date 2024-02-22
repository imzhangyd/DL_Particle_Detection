import torch.optim as optim

def func_getoptimizer(parameters, mode="SGD", lr=0.001, momentum=0.9, wd=1e-6, beta1=0.5, beta2=0.999):
    if mode == 'SGD':
        return optim.SGD(parameters, lr=lr, momentum=momentum, weight_decay=wd)
    elif mode == "Adam":
        return optim.Adam(parameters, lr=lr, betas=(beta1, beta2), weight_decay=wd)
    elif mode == "amsAdam":
        return optim.Adam(parameters, lr=lr, betas=(beta1, beta2), weight_decay=wd, amsgrad=True)