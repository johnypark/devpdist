import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="devpdist", # Replace with your own username
    version="dev0.0.1",
    author="John Park",
    author_email="parkjohnyc@gmail.com",
    description="devpdist",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/johnypark/devpdist",
    packages=setuptools.find_packages(),
    install_requires = ['tensorflow'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)

