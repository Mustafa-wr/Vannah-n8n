import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "aws-1-ap-northeast-1.pooler.supabase.com")
DB_PORT = os.getenv("DB_PORT", "6543")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres.aiphbwziyeucwabkwzih")
DB_PASS = os.getenv("DB_PASS", "FPVDEPOT!@#")

# Fix specific product_ids with NULL weights
updates = [
    (124, '3'),  # Foxeer Nano Analog Camera
    (125, '3'),  # Caddx Ant Nano Analog Camera
    (126, '3'),  # BetaFPV Nano Analog Camera
    (127, '8'),  # Foxeer Micro Analog Camera
    (128, '8'),  # Caddx Micro Analog Camera
]

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = conn.cursor()
    
    for product_id, weight in updates:
        cursor.execute('UPDATE "Products" SET weight_g = %s WHERE product_id = %s', (weight, product_id))
        print(f"[OK] Updated product_id {product_id} with weight {weight}g")
    
    conn.commit()
    
    # Verify
    cursor.execute('SELECT product_id, title, weight_g FROM "Products" WHERE product_id IN (124, 125, 126, 127, 128)')
    print("\nVerification:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} - {row[2]}g")
    
    cursor.close()
    conn.close()
    print("\n[DONE] Weights fixed!")
    
except Exception as e:
    print(f"[ERROR] {e}")
