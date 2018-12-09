#!/usr/bin/env python3
#
# Reanalysis
#

from setuptools import setup
from os import path
from io import open

package = 'rnlyss'

description = 'Python library to download, stack, and manipulate atmospheric reanalyses.'

requirements = [
    'requests',
    'numpy',
    'scipy',
    'pandas',
    'matplotlib',
    'h5py',
    'netCDF4'
]

here = path.abspath(path.dirname(__file__))

# Set the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Set version
__version__ = None
with open(path.join(here, package, '__init__.py'), encoding='utf-8') as f:
    for line in f:
        if line.startswith('__version__'):
            exec(line.strip())
            break

setup(
    name=package,
    version=__version__,
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Michael Roth',
    author_email='michael.roth@klimaat.ca',
    keywords='reanalysis weather climate',
    packages=[package],
    scripts=['bin/rnlyss_download.py', 'bin/rnlyss_stack.py'],
    install_requires=requirements,
    license='MIT',
    url=r'https://github.com/klimaat/rnlyss',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 4 - Beta",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
    ]
)
