import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="transformer_viewer", # Replace with your own username
    version="0.0.1",
    author="lucas_xing",
    author_email="xingyun44@hotmail.com",
    description="simple text visualization for transformer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Lucas0404/transformer_viewer",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
