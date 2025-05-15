from setuptools import find_packages, setup


__version__ = '1.4.44'


setup(name='abacusai',
      version=__version__,
      description='Abacus.AI Python Client Library',
      url='https://github.com/abacusai/api-python',
      author='Abacus.AI',
      author_email='dev@abacus.ai',
      license='MIT',
      packages=find_packages(),
      install_requires=['packaging', 'requests', 'pandas', 'fastavro', 'typing_inspect; python_version < "3.8"',],
      zip_safe=True,
      package_data={'': ['public.pem']},
      include_package_data=True,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Topic :: Utilities',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.8',
          'Programming Language :: Python :: 3.9',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12',
      ])
