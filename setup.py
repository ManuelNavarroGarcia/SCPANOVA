import pathlib

from setuptools import find_packages, setup

README = (pathlib.Path(__file__).parent / "README.md").read_text()

setup(
    name="scpanova",
    license="MIT",
    version="0.1.0",
    packages=find_packages(),
    long_description=README,
    long_description_content_type="text/markdown",
    author="Manuel Navarro García",
    author_email="manuelnavarrogithub@gmail.com",
    python_requires=">=3.9",
    setup_requires=["setuptools_scm"],
    install_requires=["cpsplines", "pyarrow", "rpy2", "scikit-learn", "typer"],
    extras_require={"dev": ["black", "ipykernel", "pip-tools>=7.0.0"]},
)
