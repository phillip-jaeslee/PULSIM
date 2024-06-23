from setuptools import setup, find_packages

setup(
    name='PULSIM',
    version='0.0.2',
    packages=find_packages(),
    install_requires=[
        "scipy",
        "numpy",
        "matplotlib",
        "joblib",
        "inflect",
        "ipympl",
        "torch"
    ],
    author='Jaeseok Lee',
    author_email='jslee24@snu.ac.kr',
    description='A package for pulse simulation of shaped and composite pulse',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/phillip-jaeslee/PULSIM',  # Update with your actual URL
    license="MIT",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
