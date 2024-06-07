from mat_operator import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# density matrix formalism
def thermal_eq(boltzmann_factor):
    
    return 1/2 * spin_half.unity() + 1/2 * boltzmann_factor * spin_half.Iz()

# Spin state description