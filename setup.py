import numpy
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "neuro.neuro_cy",
        ["neuro/neuro_cy.pyx"],
        libraries=["m"],
        extra_compile_args=["-O3", "-Wall"],
        language="c++",
        library_dirs=[],
        include_dirs=["neuro", numpy.get_include()]
    ),
]

setup(name='neuro',
      version='0.1',
      description='Small Python implementation of a Feed Forward Neural Network',
      #long_description=readme(),
      classifiers=[
          'Development Status :: 0 - Alpha',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Programming Language :: Python :: 2.7',
          'Topic :: Statistics :: Neural Network',
      ],
      keywords='neural network',
      url='http://github.com/',
      author='Gabriel Kabbe',
      author_email='gabriel.kabbe@chemie.uni-halle.de',
      license='GPLv3',
      install_requires=[
          'numpy',
          'ipdb',
      ],
      setup_requires = [
          'numpy'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      entry_points={
          'console_scripts': [
                             ],
      },
      include_package_data=True,
      zip_safe=False,
      ext_modules=ext_modules,
      cmdclass={'build_ext': build_ext}
      )
