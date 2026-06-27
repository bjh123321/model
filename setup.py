from setuptools import find_packages, setup


subpackages = find_packages(where=".")

setup(
    name="surrogate_model",
    version="0.1.0",
    packages=[
        "surrogate_model",
        *[f"surrogate_model.{package}" for package in subpackages],
    ],
    package_dir={"surrogate_model": "."},
)
