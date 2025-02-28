import torch


def l1(model):
    l1_norm = torch.tensor(0.).to(model.device)
    # l1_norm = l1_norm.cuda() if gpu else l1_norm

    for param in model.parameters():
        l1_norm += torch.norm(param, 1)

    return l1_norm


def l2(model):
    l2_norm = torch.tensor(0.).to(model.device)
    # l2_norm = l2_norm.cuda() if gpu else l2_norm

    for param in model.parameters():
        l2_norm += torch.norm(param)

    return l2_norm