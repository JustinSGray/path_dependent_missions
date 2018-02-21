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
      packages=[
        'path_dependent_missions',
        'path_dependent_missions/CRM',
        'path_dependent_missions/escort',
        'path_dependent_missions/simple_heat',
        ],
      install_requires=[
        'openmdao',
        'scipy',
        'pep8',
        'parameterized',
      ],
    )
