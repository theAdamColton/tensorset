from setuptools import setup

setup(
    name="tensorsequence",
    version="0.3.0",
    description="manipulate sequences of tensors",
    author="Adam Colton",
    url="https://github.com/theAdamColton/tensorsequence",
    install_requires=[
        "torch>=2.0.1",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
