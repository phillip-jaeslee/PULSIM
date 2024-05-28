import torch


## Rotation operator
# theta : radian degree

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def Rx(theta):
    return torch.stack([
        torch.stack([torch.ones_like(theta), torch.zeros_like(theta), torch.zeros_like(theta)], dim=0),
        torch.stack([torch.zeros_like(theta), torch.cos(theta), torch.sin(theta)], dim=0),
        torch.stack([torch.zeros_like(theta), -torch.sin(theta), torch.cos(theta)], dim=0)
    ], dim=0).to(device)

def Ry(theta):
    return torch.stack([
        torch.stack([torch.cos(theta), torch.zeros_like(theta), torch.sin(theta)], dim=0),
        torch.stack([torch.zeros_like(theta), torch.ones_like(theta), torch.zeros_like(theta)], dim=0),
        torch.stack([-torch.sin(theta), torch.zeros_like(theta), torch.cos(theta)], dim=0)
    ], dim=0).to(device)

def Rz(theta):
    return torch.stack([
        torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros_like(theta)], dim=0),
        torch.stack([-torch.sin(theta), torch.cos(theta), torch.zeros_like(theta)], dim=0),
        torch.stack([torch.zeros_like(theta), torch.zeros_like(theta), torch.ones_like(theta)], dim=0)
    ], dim=0).to(device)

def Rot(theta):
    return torch.stack([
        torch.stack([torch.cos(theta), torch.sin(theta)], dim=0),
        torch.stack([-torch.sin(theta), torch.cos(theta)], dim=0)
    ], dim=0).to(device)


