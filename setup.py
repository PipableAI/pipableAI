from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pipable",
    version="0.1.0",
    description="Simplify the process of connecting to remote PostgreSQL servers and executing natural language-based data search queries.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="PipableAI",
    author_email="dev@pipable.ai",
    project_urls={
        "HomePage": "https://pipable.ai/",
        "Documentation": "https://pipableai.github.io/pipable-docs/",
        "Source Code": "https://github.com/PipableAI/pipable-released",
        "Issue Tracker": "https://github.com/PipableAI/pipable-released/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "pandas>=2.0.0",
        "psycopg2-binary>=2.9.0,<=2.9.9",
        "requests>=2.28",
    ],
    python_requires=">=3.7",
)
