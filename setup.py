from setuptools import setup

setup(
    name='dotasquad',
    version='0.1',
    description='Recurrent neural network for Dota pick recommendations',
    url='https://github.com/dotasquad.git',
    author='J. Scott Moreland',
    author_email='morelandjs@gmail.com',
    packages=['dotasquad'],
    entry_points={
        'console_scripts': ['dotasquad = dotasquad.__main__:main']
        }
    )
