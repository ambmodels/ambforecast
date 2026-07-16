import glob
import os
from setuptools import setup

py_modules = [
    os.path.splitext(os.path.basename(f))[0]
    for f in glob.glob("*.py")
    if f != "setup.py"
]

setup(
    name="ambforecast",
    version="0.0.1",
    py_modules=py_modules,
    packages=["swast_forecast"],
)
