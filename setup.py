from setuptools import setup, find_packages

setup(
    name="pulsim",
    version="0.1.0",
    packages=find_packages(include=["PULSIM", "PULSIM.*"]),
    install_requires=["numpy", "scipy", "joblib"],
)