from setuptools import setup

setup(
    name="tensorset",
    version="0.4.2",
    description="manipulate sets of tensors",
    author="Adam Colton",
    url="https://github.com/theAdamColton/tensorset",
    install_requires=[
        "torch>=2.0.1",
    ],
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
)
