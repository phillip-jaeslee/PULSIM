import numpy as np
from PULSIM.backend import NumpyBackend, TorchBackend

numpy_backend = NumpyBackend()
torch_backend = TorchBackend()

M = np.array([[0, 0, 0],
              [0, 0, 0],
              [1, 1, 1]], dtype=float)
B = np.array([[1.0, 0.0, 0.10],
              [1.0, 0.0, 0.05],
              [1.0, 0.0, 0.00]])
dt = 0.006

M_numpy = numpy_backend.rotate(M, dt, B, "x")
M_torch = torch_backend.rotate(M, dt, B, "x")

print("numpy:\n", M_numpy)
print("torch:\n", M_torch)
print("agree:", np.allclose(M_numpy, M_torch, atol=1e-5))