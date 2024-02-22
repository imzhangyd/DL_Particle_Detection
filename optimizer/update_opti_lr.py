
def func_update_opti_lr(optim,epoch,decay_every):
    this_lr = optim.param_groups[0]['lr']
    change_lr = this_lr*(0.5**(epoch//decay_every))
    return change_lr