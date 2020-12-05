from setuptools import setup, find_packages

setup(
    name="FEAMAL",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "pandas"],
    extras_require={"dev": ["pytest"]},
)

#batch normalisation
#validation dataset