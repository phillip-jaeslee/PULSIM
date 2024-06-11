from mat_operator import spin_half
from spin_system import *

print("Ia",spin_half.Ia())
print("Ib",spin_half.Ib())
print("Ip",spin_half.Ip())
print("In",spin_half.In())
print("Unity",spin_half.unity())
print("Ix", spin_half.Ix())
print("Iy", spin_half.Iy())
print("Iz", spin_half.Iz())

print("spinsys", spin_system(2))