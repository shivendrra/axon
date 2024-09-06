from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext as _build_ext
import os
import codecs
import pybind11

current_dir = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(current_dir, "README.md"), encoding="utf-8") as file:
  long_description = file.read()

VERSION = '1.0.0'
DESCRIPTION = 'Multi-dimensional array creation & manipulation library like numpy written from scratch in python along with a scalar level autograd engine written in C/C++ with python wrapper'

class build_ext(_build_ext):
  def build_extensions(self):
    super().build_extensions()

# Define the C++ extension
ext_modules = [
  Extension(
    name='axon.micro.csrc.engine',
    sources=['axon/micro/csrc/engine.cpp'],
    include_dirs=['axon/micro/csrc'],
    language='c++'
  ),
]

setup(
  name="axon-pypi",
  version=VERSION,
  author="shivendra",
  author_email="shivharsh44@gmail.com",
  description=DESCRIPTION,
  long_description=long_description,
  long_description_content_type="text/markdown",
  license="MIT",
  packages=find_packages(),
  classifiers=[
    "Development Status :: 1 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Hobbyist/Enthusiasts",
    "Intended Audience :: Builders from Scratch",
    "Programming Language :: C",
    "Programming Language :: C++",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Operating System :: Windows",
    "Operating System :: MacOS",
    "Operating System :: Linux/Unix",
    "License :: OSI Approved :: MIT License",
  ],
  ext_modules=ext_modules,
  cmdclass={'build_ext': build_ext},
  entry_points={
    'console_scripts': [
      'axon=axon.__main__:main',
    ],
  },
  install_requires=[
    'pybind11>=2.10.0',
  ],
)