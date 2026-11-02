"""Database initialization script for TechGear Solutions.

Creates and seeds the SQLite database (data/techgear.db) with:
- products: Hardware inventory catalog (12 products).
- sales: Historical transactional records across geographic regions (20 records).

This script is idempotent and can be safely re-run to reset the database.
"""

import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path(__file__).resolve().parent / "techgear.db"

PRODUCTS_DATA = [
    ("SKU-NV-4090", "TechGear GeForce RTX 4090 OC 24GB", "GPU", 1799.99, 15, 36),
    ("CHG-PD-140W", "TechGear 140W GaN USB-C PD Charger", "Accessories", 89.99, 120, 12),
    ("SKU-CPU-7800X", "AMD Ryzen 7 7800X3D Desktop Processor", "CPU", 399.99, 45, 36),
    ("SKU-CPU-14900K", "Intel Core i9-14900K Desktop Processor", "CPU", 549.99, 30, 36),
    ("SKU-GPU-7900XTX", "AMD Radeon RX 7900 XTX Gaming 24GB", "GPU", 949.99, 20, 36),
    ("SKU-RAM-DDR5-32", "Corsair Vengeance 32GB (2x16GB) DDR5-6000", "Memory", 119.99, 85, 60),
    ("SKU-SSD-2TB-990", "Samsung 990 Pro 2TB PCIe 4.0 NVMe SSD", "Storage", 169.99, 60, 60),
    ("SKU-PSU-1000W", "Seasonic Vertex GX-1000 ATX 3.0 Gold PSU", "Power Supply", 199.99, 40, 120),
    ("SKU-MBD-Z790-A", "ASUS ROG Strix Z790-E Gaming WiFi Motherboard", "Motherboard", 379.99, 25, 36),
    ("SKU-MON-4K-144", "LG UltraGear 32-inch 4K UHD 144Hz Monitor", "Display", 699.99, 18, 24),
    ("SKU-CASE-AIR5000", "Corsair 5000D Airflow Tempered Glass Mid-Tower", "Chassis", 149.99, 35, 24),
    ("SKU-COOL-360AIO", "TechGear Liquid Frost 360mm AIO Cooler", "Cooling", 129.99, 50, 36),
]

SALES_DATA = [
    (1001, "SKU-NV-4090", "North", 2, 3599.98, "2026-01-05"),
    (1002, "CHG-PD-140W", "West", 10, 899.90, "2026-01-07"),
    (1003, "SKU-CPU-7800X", "Central", 4, 1599.96, "2026-01-10"),
    (1004, "SKU-RAM-DDR5-32", "East", 8, 959.92, "2026-01-12"),
    (1005, "SKU-SSD-2TB-990", "South", 5, 849.95, "2026-01-15"),
    (1006, "SKU-GPU-7900XTX", "West", 1, 949.99, "2026-01-18"),
    (1007, "SKU-PSU-1000W", "North", 3, 599.97, "2026-01-20"),
    (1008, "SKU-MBD-Z790-A", "Central", 2, 759.98, "2026-01-22"),
    (1009, "SKU-MON-4K-144", "East", 2, 1399.98, "2026-01-25"),
    (1010, "SKU-CASE-AIR5000", "South", 4, 599.96, "2026-01-28"),
    (1011, "SKU-COOL-360AIO", "West", 3, 389.97, "2026-02-01"),
    (1012, "SKU-CPU-14900K", "North", 2, 1099.98, "2026-02-03"),
    (1013, "SKU-NV-4090", "East", 1, 1799.99, "2026-02-06"),
    (1014, "CHG-PD-140W", "Central", 15, 1349.85, "2026-02-08"),
    (1015, "SKU-SSD-2TB-990", "North", 6, 1019.94, "2026-02-12"),
    (1016, "SKU-RAM-DDR5-32", "South", 12, 1439.88, "2026-02-15"),
    (1017, "SKU-GPU-7900XTX", "Central", 2, 1899.98, "2026-02-18"),
    (1018, "SKU-CPU-7800X", "West", 5, 1999.95, "2026-02-21"),
    (1019, "SKU-PSU-1000W", "East", 4, 799.96, "2026-02-24"),
    (1020, "SKU-MON-4K-144", "South", 1, 699.99, "2026-02-26"),
]


def init_database(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Initializes and populates techgear.db tables idempotently."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        # Drop existing tables to ensure clean, idempotent setup
        cursor.execute("DROP TABLE IF EXISTS sales;")
        cursor.execute("DROP TABLE IF EXISTS products;")

        # Create products table
        cursor.execute(
            """
            CREATE TABLE products (
                sku TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                category TEXT NOT NULL,
                price REAL NOT NULL,
                stock INT NOT NULL,
                warranty_months INT NOT NULL
            );
            """
        )

        # Create sales table
        cursor.execute(
            """
            CREATE TABLE sales (
                order_id INT PRIMARY KEY,
                sku TEXT NOT NULL,
                region TEXT NOT NULL,
                units INT NOT NULL,
                revenue REAL NOT NULL,
                order_date TEXT NOT NULL,
                FOREIGN KEY (sku) REFERENCES products(sku)
            );
            """
        )

        # Insert products
        cursor.executemany(
            """
            INSERT INTO products (sku, name, category, price, stock, warranty_months)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            PRODUCTS_DATA,
        )

        # Insert sales records
        cursor.executemany(
            """
            INSERT INTO sales (order_id, sku, region, units, revenue, order_date)
            VALUES (?, ?, ?, ?, ?, ?);
            """,
            SALES_DATA,
        )

        conn.commit()

        # Verify seeded data
        cursor.execute("SELECT COUNT(*) FROM products;")
        product_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM sales;")
        sales_count = cursor.fetchone()[0]

        print(f"[OK] Database successfully initialized at: {db_path}")
        print(f"     - products table: {product_count} rows seeded")
        print(f"     - sales table:    {sales_count} rows seeded")

    finally:
        conn.close()


if __name__ == "__main__":
    init_database()

