from setuptools import setup, find_packages

setup(
    name='dotasquad',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    description='Dense neural network for Dota pick recommendations',
    url='https://github.com/dotasquad.git',
    author='J. Scott Moreland',
    author_email='morelandjs@gmail.com',
    )
