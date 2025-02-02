from setuptools import setup, find_packages

setup(
    name="RGS",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "tqdm"
    ]
)