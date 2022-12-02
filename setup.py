from setuptools import setup


__version__ = '0.41.0'


setup(name='abacusai',
      version=__version__,
      description='Abacus.AI Python Client Library',
      url='https://github.com/abacusai/api-python',
      author='Abacus.AI',
      author_email='dev@abacus.ai',
      license='MIT',
      packages=['abacusai'],
      install_requires=['packaging', 'requests', 'pandas', 'fastavro'],
      zip_safe=True,
      package_data={'': ['public.pem']},
      include_package_data=True,
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Intended Audience :: Developers',
          'Topic :: Utilities',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
      ])
