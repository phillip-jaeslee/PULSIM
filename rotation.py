import numpy as np

## Rotation operator
# flip : radian degree

def Rx(flip):
    Rx = np.array ([[1, 0, 0],
                   [0, np.cos(flip), np.sin(flip)],
                   [0, -np.sin(flip), np.cos(flip)]])
    return Rx

def Ry(flip):
    Ry = np.array ([[np.cos(flip), 0, np.sin(flip)],
                   [0, 1, 0],
                   [-np.sin(flip), 0, np.cos(flip)]])
    return Ry

def Rz(flip):
    Rz = np.array ([[np.cos(flip), np.sin(flip), 0],
                   [-np.sin(flip), np.cos(flip), 0],
                   [0, 0, 1]])
    return Rz

def Rot(flip):
    Rot = np.array ([[np.cos(flip), -np.sin(flip)],
                   [np.sin(flip), np.cos(flip)]])
    return Rot

