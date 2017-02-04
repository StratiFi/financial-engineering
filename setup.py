#!/usr/bin/env python

from distutils.core import setup

setup(name='financial-engineering',
      version='1.0',
      description='Financial Engineering Utilities',
      author='DrG',
      license='StratiFi',
      author_email='guillaume@stratifi.com',
      url='https://github.com/StratiFi/financial-engineering/',
      packages=['fineng','fineng/metrics'],
      install_requires=['pykalman'],
     )
