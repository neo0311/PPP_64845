from setuptools import setup, find_packages

setup(
    name="FEAMAL",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=["numpy", "pandas", "json"],
    extras_require={"dev": ["pytest"]},
)

#batch normalisation
#validation dataset

##tests
#-LHC
#test_train_split
#construct_weights

##recently added 
#min_max_norm
#adam
#QMC sampling
#predict method
#save method (restricted for now)
#load weights method (restricted for now)