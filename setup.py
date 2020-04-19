# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

version = '0.0.1'

dependencies = [
	'tensorflow >= 1.10, < 2.0.0',
	'keras >= 2.2.2'
]

setup(
    name='keras_grid',
    version=version,
    description='Provides a wrapper for keras models to consistently try a grid of parameterized models.',
    author='Nikolai Nowaczyk',
    author_email='mail@nikno.de',
    license='MIT',
    url='https://github.com/niknow/keras-grid',
    packages=find_packages(),
    test_suite='keras_grid.tests',
    tests_require=dependencies,
    install_requires=dependencies
)
