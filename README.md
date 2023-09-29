# Pipable 🧪

# Pipable

Pipable is a Python package that provides a high-level interface for connecting to a remote PostgreSQL server, generating and executing natural language-based data search queries mapped to SQL queries using a language model.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## About

Pipable simplifies the process of querying a PostgreSQL database by allowing users to express their queries in natural language. It uses a language model to translate these natural language queries into SQL queries for the specific database and executes them on the server, returning the results in a structured format.

## Features

- **Natural Language Queries**: Express database queries in plain English.
- **PostgreSQL Integration**: Seamlessly connects to PostgreSQL databases.
- **Language Model Mapping**: Translates natural language queries into SQL queries.
- **Structured Results**: Returns query results in a structured format for easy processing.

## Installation

You can install Pipable using either the source distribution (tar.gz) or the wheel distribution (whl) available in the `dist/` directory.

### From Source Distribution (tar.gz)

```bash
pip3 install dist/pipable-<version>.tar.gz
```

Replace `<version>` with the appropriate version number of the package.

### From Wheel Distribution (whl)

```bash
pip3 install dist/pipable-<version>-py3-none-any.whl
```

Replace `<version>` with the appropriate version number of the package [current:1.0.0].

## Usage

```python
from pipable import Pipable
from pipable.core import PostgresConfig

# Initialize Pipable with PostgresConfig and LLM API base URL
postgres_config = PostgresConfig(
    host="your-postgresql-host",
    port=5432,
    database="your-database",
    user="your-username",
    password="your-password"
)
llm_api_base_url = "https://your-llm-api-url/"

# Connect to the PostgreSQL server and LLM API
pipable = Pipable(postgres_config=postgres_config, llm_api_base_url=llm_api_base_url)

# Example: Execute a natural language query
result = pipable.execute_query("Find all employees hired in the last month.")

# Process the results
for row in result:
    print(row)
```

## Documentation

For detailed usage instructions and examples, please refer to the [official documentation](https://pipableai.github.io/pipable-docs/).

## Contributing

We welcome contributions from the community! To contribute to Pipable, follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature-name`
3. Make changes and commit: `git commit -m 'Description of changes'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

Please read our [Contribution Guidelines](CONTRIBUTING.md) for more details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to add additional sections or modify the existing ones to better suit your project's needs. Providing clear and comprehensive information in your README will help others understand and contribute to your project more effectively.
