from setuptools import setup, find_packages

setup(
    name="PULSIM",
    version="1.0.0",
    description="Shaped RF pulses and NMR pulse sequences simulated with the "
                "Bloch equations and the density matrix",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Jaeseok Lee",
    url="https://github.com/phillip-jaeslee/PULSIM",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
    ],
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