import setuptools
import os, glob
from tideph.__version__ import __version__

here = os.path.abspath(os.path.dirname(__file__))

setuptools.setup(
    name="tideph", # Replace with your own username
    version=__version__,
    author="Babatunde Akinsanmi",
    author_email="tunde.akinsanmi@unige.ch",
    description="Package to model the phase curve of tidally deformed planets",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/tundeakins/tideph",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require = { "dev": [ "pytest >=3.7 "],},
)
