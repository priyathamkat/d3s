from setuptools import setup, find_packages

setup(
    name="d3s",
    version="0.1",
    description="Diffusion Dreamed Distribution Shifts",
    author="Priyatham Kattakinda",
    author_email="pkattaki@umd.edu",
    packages=find_packages(exclude=[".gitignore", "README.md", "LICENSE.md", "*.ipynb"]),
    install_requires=[
        "pyyaml",
        "numpy",
        "matplotlib",
        "torch",
        "torchvision",
        "tqdm",
        "Pillow",
        "lpips"
    ],
)