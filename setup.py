import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="mnx", # Replace with your username
    version="0",
    author="Manex Alkorta Lopetegi",
    author_email="<manexalk@gmail.com>",
    description="--------",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ManexAlkorta/mnx",
    packages=setuptools.find_packages(),
    install_requires=[
        "ase",
        "sisl",
        "spglib",
        "numpy",
        "matplotlib",
        #"crystal-toolkit",
        ],
    classifiers=[
    ],
    python_requires='>=3.6',
)
