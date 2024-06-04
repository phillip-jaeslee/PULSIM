import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class spin_half:

    # Anglular momentum operators
    def Ix():
        return torch.stack([
            torch.stack([torch.zeros(1), torch.ones(1)], dim=0),
            torch.stack([torch.ones(1), torch.zeros(1)], dim=0)
        ], dim=0, dtype=torch.complex128).to(device) / 2

    def Iy():
        return torch.stack([
            torch.stack([torch.zeros(1), torch.ones(1)], dim=0),
            torch.stack([-torch.ones(1), torch.zeros(1)], dim=0)
        ], dim=0, dtype=torch.complex128).to(device) * -1j / 2

    def Iz():
        return torch.stack([
            torch.stack([torch.ones(1), torch.zeros(1)], dim=0),
            torch.stack([torch.zeros(1), -torch.ones(1)], dim=0)
        ], dim=0, dtype=torch.complex128).to(device) / 2


    # Unity operator
    def unity():
        return torch.stack([
            torch.stack([torch.ones(1), torch.zeros(1)], dim=0),
            torch.stack([torch.zeros(1), torch.ones(1)], dim=0)
        ], dim=0, dtype=torch.complex128).to(device) / 2
    
    # Shift operators
    def Ip():
        return torch.stack([
            torch.stack([torch.zeros(1), torch.ones(1)], dim=0),
            torch.stack([torch.zeros(1), torch.zeros(1)], dim=0)
        ], dim=0, dtype=torch.complex128).to(device) / 2

    def In():
        return torch.stack([
            torch.stack([torch.zeros(1), torch.zeros(1)], dim=0),
            torch.stack([torch.ones(1), torch.zeros(1)], dim=0)
        ], dim=0, dtype=torch.complex128).to(device) / 2
    
    # Projection operators
    def Ia():
        return torch.add(Iz(), unity(), alpha=1/2)
    
    def Ib():
        return torch.add(-Iz(), unity(), alpha=1/2)
    
    # Density matrix
    def thermal_eq(omgea, T=289.3):
        h_bar = 1.05457182E-34
        kB =  1.380649E-23
        for i in range(len(omega)):
            sum_omega = -h_bar * omega(i) / kB / T 
        omega = (-h_bar * omega / kB / T) / sum_omega
        return torch.exp(omega)
    
