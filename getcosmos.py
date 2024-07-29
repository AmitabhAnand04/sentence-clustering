import csv
from azure.cosmos import CosmosClient, exceptions, PartitionKey

# Initialize the Cosmos client

client = CosmosClient(endpoint, key)

# Define the database and container
database_name = 'chatbot_log_db'
container_name = 'ConversationLog'

# Connect to the database and container
database = client.get_database_client(database_name)
container = database.get_container_client(container_name)

def get_all_values_by_column(column_name):
    try:
        # Query to fetch all unique values of the specified column
        query = f"SELECT DISTINCT c.{column_name} FROM c WHERE c.userName != 'rakesh roshan' AND c.userName != 'amitabh anand'"
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))

        values = [item[column_name] for item in items]
        return values

    except exceptions.CosmosHttpResponseError as e:
        print(f'An error occurred: {e.message}')
        return None

def save_to_csv(column_name, values, filename):
    try:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([column_name])  # Write the column name as header
            for value in values:
                writer.writerow([value])
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f'An error occurred while writing to CSV: {e}')

# Example usage
column_name = "convPrompt"
filename = "output.csv"
results = get_all_values_by_column(column_name)

if results:
    print(f"Found {len(results)} unique values.")
    save_to_csv(column_name, results, filename)
else:
    print("No values found.")
