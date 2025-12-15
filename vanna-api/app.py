"""
Vanna AI Database Query Agent
FastAPI service for natural language to SQL conversion using Groq LLM backend.
"""

import os
import json
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from openai import OpenAI
from vanna.openai import OpenAI_Chat
from vanna.chromadb import ChromaDB_VectorStore

import pandas as pd
from sqlalchemy import create_engine, text, inspect

load_dotenv()

# ============================================================================
# Configuration
# ============================================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
INTERNAL_API_KEY = os.getenv("INTERNAL_API_KEY", "changeme")

# Supabase connection
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS")

# URL-encode password to handle special characters
from urllib.parse import quote_plus
DB_PASS_ENCODED = quote_plus(DB_PASS) if DB_PASS else ""

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ============================================================================
# Custom Vanna Class with Groq Backend
# ============================================================================

class VannaGroq(ChromaDB_VectorStore, OpenAI_Chat):
    """Vanna instance using Groq's OpenAI-compatible API and ChromaDB for training."""
    
    def __init__(self, openai_client=None, model=None, chroma_path=None):
        # Initialize ChromaDB with its own config (path only)
        chroma_config = {}
        if chroma_path:
            chroma_config["path"] = chroma_path
        ChromaDB_VectorStore.__init__(self, config=chroma_config)
        
        # Initialize OpenAI with client and model
        openai_config = {
            "client": openai_client,
            "model": model,
        }
        OpenAI_Chat.__init__(self, config=openai_config)
        
        # Explicitly set client attribute (required by OpenAI_Chat methods)
        self.client = openai_client
        self.model = model


# Global Vanna instance
vn: Optional[VannaGroq] = None


def get_vanna() -> VannaGroq:
    """Get or create Vanna instance."""
    global vn
    if vn is None:
        # Create Groq client (OpenAI-compatible)
        groq_client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1"
        )
        
        vn = VannaGroq(
            openai_client=groq_client,
            model=GROQ_MODEL,
            chroma_path="/app/chroma_data"
        )
        # Connect to Supabase
        vn.run_sql = run_sql
        vn.run_sql_is_set = True
    return vn


def run_sql(sql: str) -> pd.DataFrame:
    """Execute SQL query against Supabase and return DataFrame."""
    engine = create_engine(DATABASE_URL)
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        df = pd.DataFrame(result.fetchall(), columns=result.keys())
    return df


# ============================================================================
# API Models
# ============================================================================

class AskRequest(BaseModel):
    question: str
    execute: bool = True  # If False, only generate SQL without executing


class AskResponse(BaseModel):
    question: str
    sql: Optional[str] = None
    data: Optional[list] = None
    error: Optional[str] = None


class TrainResponse(BaseModel):
    message: str
    tables_trained: list
    sample_queries_added: int


# ============================================================================
# Authentication
# ============================================================================

async def verify_api_key(x_api_key: str = Header(None)):
    """Simple API key authentication."""
    if x_api_key != INTERNAL_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ============================================================================
# FastAPI App
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize Vanna on startup."""
    get_vanna()
    yield


app = FastAPI(
    title="Vanna Database Query Agent",
    description="Natural language to SQL for shopping drone database",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker/n8n."""
    return {"status": "healthy", "model": GROQ_MODEL}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
async def ask_question(request: AskRequest):
    """
    Convert natural language question to SQL and optionally execute it.
    
    Example questions:
    - "Show me drones under $500 with weight less than 250g"
    - "What are the top 10 best selling drones?"
    - "List all drones with pink color options"
    """
    try:
        vanna = get_vanna()
        
        # Generate SQL from natural language
        sql = vanna.generate_sql(request.question)
        
        if not sql:
            return AskResponse(
                question=request.question,
                error="Could not generate SQL for this question"
            )
        
        # Clean up SQL (remove markdown formatting if present)
        sql = sql.strip()
        if sql.startswith("```"):
            sql = sql.split("\n", 1)[1] if "\n" in sql else sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()
        
        data = None
        if request.execute:
            try:
                df = run_sql(sql)
                data = df.to_dict(orient="records")
            except Exception as e:
                return AskResponse(
                    question=request.question,
                    sql=sql,
                    error=f"SQL execution failed: {str(e)}"
                )
        
        return AskResponse(
            question=request.question,
            sql=sql,
            data=data
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainResponse, dependencies=[Depends(verify_api_key)])
async def train_schema():
    """
    Train Vanna on the database schema.
    Auto-discovers all tables and their structures from Supabase.
    Run this after your database schema is set up.
    """
    try:
        vanna = get_vanna()
        engine = create_engine(DATABASE_URL)
        inspector = inspect(engine)
        
        tables_trained = []
        
        # Get all tables in public schema
        for table_name in inspector.get_table_names(schema="public"):
            # Get CREATE TABLE statement (DDL)
            columns = inspector.get_columns(table_name, schema="public")
            pk = inspector.get_pk_constraint(table_name, schema="public")
            fks = inspector.get_foreign_keys(table_name, schema="public")
            
            # Build DDL string
            col_defs = []
            for col in columns:
                col_def = f"  {col['name']} {col['type']}"
                if not col.get('nullable', True):
                    col_def += " NOT NULL"
                col_defs.append(col_def)
            
            # Add primary key
            if pk and pk.get('constrained_columns'):
                col_defs.append(f"  PRIMARY KEY ({', '.join(pk['constrained_columns'])})")
            
            ddl = f"CREATE TABLE {table_name} (\n" + ",\n".join(col_defs) + "\n);"
            
            # Train Vanna on this DDL
            vanna.train(ddl=ddl)
            tables_trained.append(table_name)
            
            # Add foreign key relationships as documentation
            for fk in fks:
                fk_doc = (
                    f"Table {table_name} has a foreign key relationship: "
                    f"{', '.join(fk['constrained_columns'])} references "
                    f"{fk['referred_table']}({', '.join(fk['referred_columns'])})"
                )
                vanna.train(documentation=fk_doc)
        
        # Add some sample queries for shopping drone context
        # Note: PostgreSQL requires double-quotes for mixed-case table names
        sample_queries = [
            {
                "question": "Show me all products",
                "sql": 'SELECT * FROM "Products"'
            },
            {
                "question": "Show me all drones under $500",
                "sql": 'SELECT * FROM "Products" WHERE price < 500 ORDER BY price ASC'
            },
            {
                "question": "List drones with weight less than 250 grams",
                "sql": 'SELECT * FROM "Products" WHERE weight_g < 250 ORDER BY weight_g ASC'
            },
            {
                "question": "How many products are there",
                "sql": 'SELECT COUNT(*) FROM "Products"'
            },
            {
                "question": "Show me all product categories",
                "sql": 'SELECT * FROM product_category'
            },
            {
                "question": "What motor specs are available",
                "sql": 'SELECT * FROM motor_specs'
            }
        ]
        
        # Add documentation about table naming convention
        vanna.train(documentation="""
            DATABASE SCHEMA GUIDE:
            
            1. MAIN PRODUCTS TABLE: "Products" (capital P, requires double quotes)
               - Contains: product_id, title, brand, category, sub_category, weight_g, status, etc.
               - USE THIS TABLE for questions about:
                 * Product listings, filtering, searching
                 * Weight queries (weight_g column)
                 * Category/brand filtering
                 * General product information
               - Example: SELECT * FROM "Products" WHERE weight_g < 250
               - Example: SELECT * FROM "Products" WHERE category = 'motor'
            
            2. SPEC TABLES (lowercase, no quotes needed):
               - motor_specs: Technical motor specifications (kv, stator size, etc.)
               - battery_specs: Battery specifications (capacity, voltage, etc.)
               - frame_specs: Frame specifications
               - camera_specs: Camera specifications
               - USE THESE for detailed technical specifications, NOT for weight/price queries
            
            3. IMPORTANT RULES:
               - Always use double quotes for "Products": SELECT * FROM "Products"
               - For product weight queries, use "Products".weight_g
               - For product counts/listings, use "Products"
               - Join spec tables with "Products" for combined queries
        """)
        
        for sq in sample_queries:
            try:
                vanna.train(question=sq["question"], sql=sq["sql"])
            except:
                pass  # Ignore if tables don't exist yet
        
        return TrainResponse(
            message="Training completed successfully",
            tables_trained=tables_trained,
            sample_queries_added=len(sample_queries)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tables", dependencies=[Depends(verify_api_key)])
async def list_tables():
    """List all tables in the database."""
    try:
        engine = create_engine(DATABASE_URL)
        inspector = inspect(engine)
        tables = inspector.get_table_names(schema="public")
        return {"tables": tables}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
