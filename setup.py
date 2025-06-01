from setuptools import setup, find_packages

# Read requirements from requirements.txt
try:
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = []

setup(
    name="smart-manufacturing-machines-efficiency-prediction",
    version="0.1",
    author="Faheem Khan",
    author_email="faheemthakur23@gmail.com",
    description="End-to-End MLOps Project for Smart Manufacturing Machines Efficiency Prediction",
    packages=find_packages(),
    install_requires=requirements
)