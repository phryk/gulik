#!/usr/bin/env python-3.6

from distutils.core import setup

setup(  name='nukular',
        description='An advanced graphical system monitoring suite using cairo.',
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
        scripts=['nukular']
    )
