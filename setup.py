import os
import sys
from distutils.core import setup
from setuptools import find_packages

setup(name='path_dependent_missions',
      version='0.1',
      description="Cases for mission optimization",
      long_description="""\
""",
      classifiers=[
        'Development Status :: 2 - Pre-Alpha',
      ],
      keywords='',
      author='John Jasa',
      author_email='johnjasa@umich.edu',
      license='Apache License, Version 2.0',
      packages=find_packages(),
      install_requires=[
        'openmdao',
        'numpy>=1.9.2',
        'scipy',
        'pep8',
        'parameterized',
      ],
    )
