import setuptools

def get_readme():
    with open("README.md", "r") as f:
        return f.read()


setuptools.setup(
    name="coordination",
    version="0.1.0",
    description="Symmetry, Equilibria, and Robustness in Common-Payoff Games",
    long_description=get_readme(),
    long_description_content_type="text/markdown",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scipy",
        "sympy",
    ],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
    ]
)
