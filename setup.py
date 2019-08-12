from distutils.core import setup

setup(
    name='ieee',
    author='Piotr Gabryś',
    author_email='piotrek.gabrys@gmail.com',
    version='0.0.1',
    packages=['ieee',],
    install_requires=[
        'scikit-learn',
        'pandas',
        'numpy'],
    license='MIT',
    long_description=open('README.md').read(),
    url="https://github.com/PiotrekGa/ieee"
)