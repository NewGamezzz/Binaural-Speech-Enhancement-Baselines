def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
