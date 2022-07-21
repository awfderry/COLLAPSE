from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='collapse',
    packages=find_packages(include=[
        'collapse',
        'collapse.data',
        'collapse.models',
        'collapse.utils',
        'collapse.atom_info',
        'collapse.byol_pytorch'
    ]),
    version='0.1.0',
    description='COLLAPSE',
    license='MIT',
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'torch',
        'torch_geometric',
        'torch_scatter',
        'numpy',
        'pandas',
        'atom3d',
        'biopython',
        'scipy'
    ]
)
