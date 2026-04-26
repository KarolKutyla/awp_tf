from setuptools import setup, find_packages

setup(
    name="awp_protocol",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "adversarial-robustness-toolbox",
    ],
)