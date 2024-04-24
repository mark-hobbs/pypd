from setuptools import setup, find_packages

setup(
    name="PyPD",
    version="0.0.1",
    description="A simple bond-based peridynamics code written in Python",
    long_description="Long description of your package",
    author="Mark Hobbs",
    author_email="markhobbs91@gmail.com",
    url="https://github.com/mark-hobbs/PyPD",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "numba",
        "matplotlib",
        "scikit-learn",
        "tqdm",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
