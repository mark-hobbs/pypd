from setuptools import setup, find_packages

setup(
    name='package_name',
    version='1.0.0',
    description='Description of your package',
    long_description='Long description of your package',
    author='Your Name',
    author_email='your@email.com',
    url='https://github.com/yourusername/package_name',
    packages=find_packages(),
    install_requires=[
        'dependency1',
        'dependency2',
        # Add any additional dependencies here
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)