import psycopg2
import sys

# Database credentials
host = '34.82.20.16'
username = 'pipable'
password = 'pipable'
database = 'postgres'
schema = 'alyf'
table_name = 'alyf_asvs'

try:
    # Connect to the database
    print("Connecting to the PostgreSQL database...")
    connection = psycopg2.connect(
        host=host,
        user=username,
        password=password,
        database=database
    )
    print("Connection successful!")

    # Create a cursor to interact with the database
    cursor = connection.cursor()

    # Set the search path to the desired schema
    cursor.execute(f"SET search_path TO {schema}")

    # Execute a simple select statement
    cursor.execute(f"SELECT * FROM {table_name}")

    # Fetch all rows from the result
    rows = cursor.fetchall()

    # Print the rows
    for row in rows:
        print(row)

    # Close the cursor and connection
    cursor.close()
    connection.close()

except (Exception, psycopg2.Error) as error:
    print("Error connecting to PostgreSQL:", error)