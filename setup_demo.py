"""Create a sample SQLite database for testing TalkDB."""
import sqlite3
from pathlib import Path


def create_sample_db(db_path: str = "demo.db") -> str:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            city TEXT NOT NULL,
            signup_date TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL,
            stock INTEGER NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            quantity INTEGER NOT NULL,
            total REAL NOT NULL,
            order_date TEXT NOT NULL,
            FOREIGN KEY (customer_id) REFERENCES customers(id),
            FOREIGN KEY (product_id) REFERENCES products(id)
        )
    """)

    customers = [
        (1, "Alice Johnson", "alice@email.com", "New York", "2025-01-15"),
        (2, "Bob Smith", "bob@email.com", "San Francisco", "2025-02-20"),
        (3, "Charlie Brown", "charlie@email.com", "Chicago", "2025-03-10"),
        (4, "Diana Ross", "diana@email.com", "New York", "2025-04-05"),
        (5, "Eve Wilson", "eve@email.com", "San Francisco", "2025-05-18"),
        (6, "Frank Miller", "frank@email.com", "Chicago", "2025-06-22"),
        (7, "Grace Lee", "grace@email.com", "Boston", "2025-07-30"),
        (8, "Henry Davis", "henry@email.com", "Boston", "2025-08-14"),
        (9, "Ivy Chen", "ivy@email.com", "New York", "2025-09-01"),
        (10, "Jack Taylor", "jack@email.com", "San Francisco", "2025-10-12"),
    ]

    products = [
        (1, "Laptop Pro", "Electronics", 1299.99, 50),
        (2, "Wireless Mouse", "Electronics", 29.99, 200),
        (3, "Desk Chair", "Furniture", 349.99, 75),
        (4, "Monitor 27inch", "Electronics", 449.99, 100),
        (5, "Keyboard Mechanical", "Electronics", 89.99, 150),
        (6, "Standing Desk", "Furniture", 599.99, 30),
        (7, "Webcam HD", "Electronics", 69.99, 120),
        (8, "Headphones", "Electronics", 199.99, 80),
        (9, "Desk Lamp", "Furniture", 45.99, 200),
        (10, "USB Hub", "Electronics", 39.99, 300),
    ]

    orders = [
        (1, 1, 1, 1, 1299.99, "2025-06-01"),
        (2, 2, 3, 2, 699.98, "2025-06-05"),
        (3, 3, 2, 3, 89.97, "2025-06-10"),
        (4, 1, 4, 1, 449.99, "2025-06-15"),
        (5, 4, 5, 2, 179.98, "2025-07-01"),
        (6, 5, 6, 1, 599.99, "2025-07-05"),
        (7, 2, 8, 1, 199.99, "2025-07-10"),
        (8, 6, 1, 1, 1299.99, "2025-07-15"),
        (9, 7, 7, 2, 139.98, "2025-07-20"),
        (10, 3, 9, 3, 137.97, "2025-07-25"),
        (11, 8, 10, 5, 199.95, "2025-08-01"),
        (12, 9, 2, 1, 29.99, "2025-08-05"),
        (13, 10, 3, 1, 349.99, "2025-08-10"),
        (14, 1, 5, 1, 89.99, "2025-08-15"),
        (15, 4, 8, 2, 399.98, "2025-08-20"),
        (16, 5, 4, 1, 449.99, "2025-09-01"),
        (17, 6, 2, 4, 119.96, "2025-09-05"),
        (18, 7, 1, 1, 1299.99, "2025-09-10"),
        (19, 8, 6, 1, 599.99, "2025-09-15"),
        (20, 9, 3, 1, 349.99, "2025-09-20"),
    ]

    cursor.executemany("INSERT OR REPLACE INTO customers VALUES (?,?,?,?,?)", customers)
    cursor.executemany("INSERT OR REPLACE INTO products VALUES (?,?,?,?,?)", products)
    cursor.executemany("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?,?)", orders)

    conn.commit()
    conn.close()
    print(f"Demo database created at: {Path(db_path).resolve()}")
    print("  - 10 customers")
    print("  - 10 products (Electronics + Furniture)")
    print("  - 20 orders")
    return db_path


if __name__ == "__main__":
    create_sample_db()
