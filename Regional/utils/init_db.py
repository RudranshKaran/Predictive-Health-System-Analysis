"""Database initialization module."""
from utils.db_operations import db_ops

def init_database():
    """Initialize the database with test data."""
    try:
        db_ops.load_initial_data()
        print("Database initialized successfully with test data.")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")

if __name__ == "__main__":
    init_database()