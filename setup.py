from setuptools import setup, find_packages

setup(
    name="awp_protocol",
    version="0.3.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "keras-hub",
        "keras-cv",
        "tensorflow-datasets",
        "importlib-resources"
    ],
)