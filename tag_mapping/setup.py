from setuptools import setup, find_packages

setup(
    name='tag_mapping',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'recognize-anything',
    ],
    description='Package for building spatial maps from image tags',
)