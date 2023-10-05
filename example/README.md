# Running the Pipable LLM Server and Query Example

This guide outlines how to run the provided Pipable LLM server and execute queries using the `example_as_query.py` script. The Pipable LLM server generates text based on the given context and question. Follow the steps below to set up and run the server, connect to a PostgreSQL database, and execute queries.

## Prerequisites

Before you begin, ensure you have the following software installed on your system:

- Python 3.7
- PostgreSQL (with the sample database installed)
- Pip3 (Python package installer)

## Setting Up the Pipable LLM Server

1. **Install Dependencies:**

    ```bash
    pip3 install fastapi uvicorn
    ```

2. **Download and Prepare the Sample Database:**

   - Download the DVD Rental Sample Database from [here](https://www.postgresqltutorial.com/wp-content/uploads/2019/05/dvdrental.zip).
   - Extract the downloaded file to obtain a file named `dvdrental.tar`.
   - Restore the database using the following command:

    ```bash
    pg_restore --dbname=sampleDB --username=postgres --file=dvdrental.tar
    ```

3. **Run the Pipable LLM Server:**

    ```bash
    uvicorn sample_llm_server:app --reload
    ```

   The server will start and listen for requests at `http://localhost:8000`.

## Running the Query Example

1. **Configure Database Connection:**

   Open `example_ask_and_execute_query.py` and update the PostgreSQL configuration parameters (host, port, database name, user, and password) to match your local PostgreSQL setup.

2. **Execute the Example Query:**

    ```bash
    python example_as_query.py
    ```

    This script will generate an SQL query using the Pipable LLM server and execute it against the specified PostgreSQL database. The query result will be printed to the console.

3. **Review Query Result:**

   After running the script, review the query result displayed in the console.

4. **Clean Up:**

   After executing the queries, make sure to disconnect from the database and stop the Pipable LLM server if no longer needed.

## Notes

- Ensure that the PostgreSQL server is running and accessible before running the Pipable LLM server and the query script.
- Modify the table names and query question in `example_as_query.py` according to your use case.