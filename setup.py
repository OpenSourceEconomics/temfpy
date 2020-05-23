from setuptools import find_packages
from setuptools import setup

p = find_packages()
print(p)

setup(
    name="temfpy",
    version="0.0",
    description=("Python package with test functions for various numerical components"),
    license="MIT",
    url="https://github.com/OpenSourceEconomics/temfpy",
    author="OpenSourceEconomics",
    author_email="eisenhauer@policy-lab.org",
    packages=p,
    zip_safe=False,
    package_data={"utilities": []},
    include_package_data=True,
)
