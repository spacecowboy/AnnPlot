#!/usr/bin/env python

from distutils.core import setup, Extension
#import numpy

#numpy_include = numpy.get_include()

#network_ext = Extension('fast_network',
#          sources = ['ann/nodemodule/fast_network.c', 'ann/nodemodule/activation_functions.c'],
#          include_dirs = [numpy_include],
#          extra_compile_args = ['-std=c99'])

setup(name = 'AnnPlot',
      version = '0.1',
      description = 'A package with functions that plot pretty figures for machine learning.',
      author = 'Jonas Kalderstam',
      author_email = 'jonas@kalderstam.se',
      url = 'https://github.com/spacecowboy/AnnPlot',
      packages = ['annplot'],
      package_dir = {'': ''},
      #ext_package = 'ann.nodemodule',
      #ext_modules = [network_ext],
      requires = ['numpy', 'matplotlib', 'sklearn'],
     )
