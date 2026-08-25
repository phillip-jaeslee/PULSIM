from PULSIM.spin_operators import SpinOperators, Ix, Iy, Iz, embed, product_operator
from PULSIM.spin_system import SpinSystem


J_HZ = 0
ss = SpinSystem(['H', '13C'], offsets=[0, 0], couplings={(0, 1): J_HZ})

ops = SpinOperators(spin_system=ss)

sigma = product_operator(Iz(), 0, Iz(), 1, 2)

print(ops.readout(sigma, ['Ix', 'Iy', 'Iz', 'IzSz']))