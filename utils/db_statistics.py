import json
import os
import dotenv
import pandas as pd
import csv

dotenv.load_dotenv()


def custom_csv_reader(table_name: str):
    """Custom CSV parser to handle an edge case in the CSV files."""
    path = os.getenv("TABLES_PATH")
    csv_path = f"{path}/{table_name}.csv"
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing file: {table_name}.csv")
    with open(csv_path, "r", encoding="utf-8", newline="") as csvfile:
        file = csvfile.read()
        lines = [
            line if r"\," not in line else line.replace("\\", '"')
            for line in file.splitlines()
        ]
        reader = csv.reader(lines, delimiter=",", quotechar='"', escapechar="\\")
        return reader


table_names = [
    os.path.splitext(f)[0]
    for f in os.listdir(os.getenv("TABLES_PATH"))
    if f.endswith(".csv")
]
row_counts = {}
for table in table_names:
    path = os.getenv("TABLES_PATH")
    csv_path = f"{path}/{table}.csv"
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        data = f.read()
        lines = data.splitlines()
        row_counts[table] = len(lines)

with open("data/row_counts.json", "w") as f:
    json.dump(row_counts, f)

with open("data/schema.json", "r") as f:
    schema = json.load(f)


unique_vals = {}
for key, value in schema.items():
    reader = custom_csv_reader(key)
    data = list(reader)
    df = pd.DataFrame(data, columns=value)
    unique_vals[key] = json.loads(df.nunique().to_json())

with open("data/unique_vals.json", "w") as f:
    json.dump(unique_vals, f)
