from setuptools import setup, find_packages

setup(
    name='tower-jump-analyzer',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A tool for analyzing tower jumps and location data.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/neiev/TowerJumpAnalyzer',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas',
        # Add other dependencies here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)