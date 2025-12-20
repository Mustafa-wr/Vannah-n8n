"""
Supabase Database Analysis Script
Connects directly to Supabase and analyzes all tables, row counts, and structure
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "aws-1-ap-northeast-1.pooler.supabase.com")
DB_PORT = os.getenv("DB_PORT", "6543")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres.aiphbwziyeucwabkwzih")
DB_PASS = os.getenv("DB_PASS", "FPVDEPOT!@#")

try:
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )
    cursor = conn.cursor()
    
    print("="*70)
    print("SUPABASE DATABASE ANALYSIS")
    print("="*70)
    
    # Get all tables
    cursor.execute("""
        SELECT table_name, table_type 
        FROM information_schema.tables 
        WHERE table_schema = 'public' 
        ORDER BY table_type, table_name
    """)
    tables = cursor.fetchall()
    
    print(f"\nTotal objects in public schema: {len(tables)}")
    
    # Separate tables and views
    base_tables = [t for t in tables if t[1] == 'BASE TABLE']
    views = [t for t in tables if t[1] == 'VIEW']
    
    print(f"  - Base Tables: {len(base_tables)}")
    print(f"  - Views: {len(views)}")
    
    # Analyze each table
    print("\n" + "="*70)
    print("BASE TABLES (with row counts)")
    print("="*70)
    
    total_rows = 0
    table_sizes = []
    
    for table_name, _ in base_tables:
        try:
            cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            count = cursor.fetchone()[0]
            total_rows += count
            table_sizes.append((table_name, count))
            
            # Get column count
            cursor.execute(f"""
                SELECT COUNT(*) FROM information_schema.columns 
                WHERE table_name = '{table_name}' AND table_schema = 'public'
            """)
            col_count = cursor.fetchone()[0]
            
            print(f"  {table_name:40} | {count:6} rows | {col_count:2} columns")
        except Exception as e:
            print(f"  {table_name:40} | ERROR: {str(e)[:30]}")
    
    print(f"\n  TOTAL ROWS: {total_rows:,}")
    
    # Views
    print("\n" + "="*70)
    print("VIEWS")
    print("="*70)
    for view_name, _ in views:
        print(f"  {view_name}")
    
    # Estimate storage size
    print("\n" + "="*70)
    print("STORAGE ESTIMATES")
    print("="*70)
    
    # Rough estimate: ~500 bytes per row average
    estimated_size_mb = (total_rows * 500) / (1024 * 1024)
    print(f"  Estimated data size: ~{estimated_size_mb:.2f} MB")
    print(f"  Total rows: {total_rows:,}")
    
    # Check for indexes
    cursor.execute("""
        SELECT COUNT(*) FROM pg_indexes WHERE schemaname = 'public'
    """)
    index_count = cursor.fetchone()[0]
    print(f"  Indexes: {index_count}")
    
    # Products table detail
    print("\n" + "="*70)
    print("PRODUCTS TABLE DETAIL")
    print("="*70)
    
    cursor.execute('SELECT category, COUNT(*) FROM "Products" GROUP BY category ORDER BY COUNT(*) DESC')
    print("\n  By Category:")
    for row in cursor.fetchall():
        print(f"    {row[0]:20}: {row[1]}")
    
    cursor.execute('SELECT brand, COUNT(*) FROM "Products" GROUP BY brand ORDER BY COUNT(*) DESC LIMIT 10')
    print("\n  Top 10 Brands:")
    for row in cursor.fetchall():
        print(f"    {row[0]:20}: {row[1]}")
    
    cursor.close()
    conn.close()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
except Exception as e:
    print(f"[ERROR] {e}")
