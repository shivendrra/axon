from setuptools import setup, find_packages
import codecs
import os

current_dir = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(current_dir, "README.md"), encoding="utf-8") as file:
  long_description = "\n" + file.read()

VERSION = '1.1.8'
DESCRIPTION = 'Multi-dimensional array creation & manipulation library like numpy written from scratch in python along with a scalar level autograd engine written in c/c++ with python wrapper'

setup(
  name="axon",
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
)