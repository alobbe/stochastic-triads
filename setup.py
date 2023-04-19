from setuptools import setup, find_packages
import datetime

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

now = datetime.datetime.now()
version = f"{now.year}.{now.month}.{now.day}"

setup(
    name="stochastic_triads",
    version=version,
    author="Alexander Lobbe",
    author_email="alex.lobbe@imperial.ac.uk",
    description="Simulate Stochastic Triad Turbulence Models",
    python_requires='>=3.10',
    # Fixed version of JAX and JAXLIB to avoid breaking changes
    install_requires=[
        "jaxlib==0.4.1",
        "jax==0.4.1",
    ],
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Fluid Dynamics",
        "Topic :: Scientific/Engineering :: Data Assimilation",
    ],
)
