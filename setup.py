from setuptools import setup, find_packages
from pathlib import Path

VERSION = '0.0.4'
DESCRIPTION = 'Python implementation of the quantum optimal control GOAT-algorithm'
# read the contents of the README file
this_directory = Path(__file__).parent
LONG_DESCRIPTION = (this_directory / "README.md").read_text()

# Setting up
setup(
    name="goat_quantumcontrol",
    version=VERSION,
    author="Markus Plautz",
    author_email="<markus.plautz@gmx.at>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=LONG_DESCRIPTION,
    url='https://github.com/pmarkus-github/goat-quantumcontrol',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib'],
    keywords=['python', 'quantum-control', 'goat', 'quantum-gates', 'quantum-computer'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)