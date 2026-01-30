"""
Database Manager Module
Handles SQLite database operations for storing performance data.
"""

import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / 'data' / 'performance.db'

def init_db():
    """Initialize the database and create tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS performance_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            cpu_percent REAL,
            cpu_freq REAL,
            memory_percent REAL,
            memory_used_gb REAL,
            disk_percent REAL,
            disk_used_gb REAL,
            net_sent_mb REAL,
            net_recv_mb REAL
        )
    ''')

    conn.commit()
    conn.close()

def store_data(df):
    """
    Store performance data in the database.

    Args:
        df (pd.DataFrame): DataFrame with performance data
    """
    conn = sqlite3.connect(DB_PATH)
    df.to_sql('performance_data', conn, if_exists='append', index=False)
    conn.close()

def load_data(limit=None):
    """
    Load performance data from the database.

    Args:
        limit (int, optional): Maximum number of records to load

    Returns:
        pd.DataFrame: Loaded data
    """
    conn = sqlite3.connect(DB_PATH)
    query = 'SELECT * FROM performance_data'
    if limit:
        query += f' LIMIT {limit}'
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

if __name__ == '__main__':
    init_db()
    print("Database initialized.")