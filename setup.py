#!/usr/bin/env python-3.6

from distutils.core import setup

setup(  name='gulik',
        description='advanced graphical system monitoring suite using cairo and psutil.',
        author='phryk',
        classifiers=[
            'Development Status :: 2 - Pre-Alpha',
            'Programming Language :: Python :: 3.6',
        ],
        install_requires=[
            'psutil',
            'pycairo',
            'PyGObject'
        ],
        scripts=['bin/gulik'],
        packages=['gulik'],
        package_data={'gulik': ['gulik.png']}
    )
