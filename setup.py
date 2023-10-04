from setuptools import find_packages, setup

setup(
    name="pipable",
    version="1.0.0",
    description="Simplify the process of connecting to remote PostgreSQL servers and executing natural language-based data search queries.",
    long_description="""Pipable is a Python package designed to simplify the process of connecting to remote PostgreSQL servers and executing natural language-based data search queries. Powered by a state-of-the-art language model, Pipable translates user-friendly search queries into SQL commands and executes them, making data retrieval and analysis effortless. With a user-friendly interface and robust backend, Pipable is the ideal tool for data analysts, developers, and anyone working with complex databases.""",
    long_description_content_type="text/plain",
    author="PipableAI",
    author_email="dev@pipable.ai",
    url="https://github.com/PipableAI/pipable-released",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=["pandas", "psycopg2"],
)
