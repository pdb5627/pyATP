#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    # TODO: put package requirements here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='pyATP',
    version='0.1.0',
    description="Module for working with ATP data files, running ATP from "
                "a Python environment, and extracting results. Also includes "
                "some line impedance and optimization functions that "
                "eventually need to be factored out.",
    long_description=readme + '\n\n' + history,
    author="Paul David Brown",
    author_email='pdb.lists@gmail.com',
    url='https://github.com/pdb5627/pyATP',
    packages=[
        'pyATP',
        'lineZ'
    ],
    package_dir={'pyATP': 'pyATP',
                 'lineZ': 'lineZ'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='ATP',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
