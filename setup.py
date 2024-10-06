from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="weave",
    version="0.1.0",
    author="Ashik Shaffi",
    author_email="ashikshaffi0@gmail.com",
    description="A flexible framework for generating and validating synthetic data across various domains",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ashikshafi08/weave",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=[
        "PyYAML>=6.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "matplotlib>=3.4.2",
        "tqdm>=4.61.1",
        "openai>=0.27.2",
        "vllm>=0.4.2",
        "transformers>=4.40.2",
        "torch>=2.3.1",
    ],
)