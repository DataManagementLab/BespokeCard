from abc import ABC, abstractmethod
import csv
import os
import dotenv
import json

dotenv.load_dotenv()


class base_card_estimator(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def setup(self):
        pass

    @abstractmethod
    def estimate(self, tables: list, filters: list, joins: list) -> int:
        pass

    def custom_csv_reader(self, table_name: str) -> csv.reader:
        """Custom CSV parser to handle an edge case in the CSV files."""
        with open("data/schema.json", "r") as f:
            schema = json.load(f)
        header = ",".join(schema[table_name].keys())
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
            lines = [header] + lines
            reader = csv.reader(lines, delimiter=",", quotechar='"', escapechar="\\")
            return reader
