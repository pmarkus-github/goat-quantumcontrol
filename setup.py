from setuptools import setup, find_packages

VERSION = '0.0.3'
DESCRIPTION = 'Python implementation of the quantum optimal control GOAT-algorithm'
LONG_DESCRIPTION = 'A package that allows you to use the GOAT-algorithm for the implementation of unitary gates' \
                   'in quantum systems.'

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
    install_requires=['numpy==1.21.5', 'scipy==1.7.3', 'matplotlib==3.5.1'],
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