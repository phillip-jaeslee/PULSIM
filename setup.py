from setuptools import setup, find_packages

setup(
    name="PULSIM",
    version="0.1.0",
    packages=find_packages(include=["PULSIM", "PULSIM.*"]),
    install_requires=["numpy>=1.25", "scipy"],
    extras_require={
        "file":     ["pandas"],         # CompositeCSVShape._read
        "torch":    ["torch"],          # TorchBackend, torch_bloch_rotate
        "parallel": ["joblib"],         # parallel_map
        "viz":      ["matplotlib", "ipywidgets"],
        "all":      ["pandas", "torch", "joblib", "matplotlib", "ipywidgets"],
    },
    python_requires=">=3.9"
)