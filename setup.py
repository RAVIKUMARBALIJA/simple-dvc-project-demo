from setuptools import setup,find_packages

with open("README.md","r",encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.1",
    author="ravikumar",
    long_description=long_description,
    description="small package for dvc demo",
    long_description_content_type="text/markdown",
    url="https://github.com/RAVIKUMARBALIJA/simple-dvc-project-demo.git",
    author_email="bravikumar123@hotmail.com",
    packages=["src"],
    license="GNU",
    python_requires=">=3.6",
    install_requires=[
        "dvc",
        "json",
        "dvc[gdrive]",
        "dvc[s3]",
        "pandas",
        "scikit-learn"
    ]

)
