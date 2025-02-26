from setuptools import setup, find_packages

setup(
    name='neonormalcb',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "pymc>=5.0",
        "bambi>=0.13",
        "numpy<2.1.0,>=1.26.0",
        "scipy",       
        "pytensor"
    ]
)