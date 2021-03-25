from setuptools import setup, find_packages

setup(
    name="PYANNET",
    version="0.9.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "pandas"],
    extras_require={"dev": ["pytest"]},
)

#batch normalisation
#validation dataset


#next commit
# new data handling function 
#NEW EVALUATION FUNCTION
#NEW NAME
#CODE CLEAN