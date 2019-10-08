import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mil-pytorch",
    version="0.0.1",
    author="Jakub Monhart",
    author_email="monhajak@fel.cvut.cz",
    description="Model for solving multiple instance learning problem implemented using pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jakubmonhart/mil_pytorch",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    install_requires = [
        'torch',
        'scikit-learn'
    ]

)