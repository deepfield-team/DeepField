"""A framework for reservoir simulation."""
import re
from setuptools import setup, find_packages

with open('deepfield/__init__.py', 'r') as f:
    VERSION = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if not VERSION:
        raise RuntimeError("Unable to find version string.")
    VERSION = VERSION.group(1)

with open('docs/index.rst', 'r') as f:
    LONG_DESCRIPTION = f.read()

setup(
    name='DeepField',
    packages=find_packages(exclude=['docs']),
    version=VERSION,
    url='https://github.com/deepfield-team/DeepField',
    license='Apache License 2.0',
    author='deepfield-team',
    author_email='',
    description='A framework for reservoir simulation',
    long_description=LONG_DESCRIPTION,
    zip_safe=False,
    platforms='any',
    install_requires=[
        'chardet',
	'pandas',
	'scikit-learn',
	'scikit-image',
	'h5py',
	'anytree',
	'ipywidgets',
	'pyvista',
	'tqdm',
	'numba',
	'tables',
	'pytest',
	'psutil',
    'dask'
    ],
    extras_require={
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: Apache License 2.0',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering'
    ],
)
