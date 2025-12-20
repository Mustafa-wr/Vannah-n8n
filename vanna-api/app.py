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

from urllib.parse import quote_plus

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

# URL-encode password to handle special characters (!@# etc.)
DB_PASS_ENCODED = quote_plus(DB_PASS) if DB_PASS else ""
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASS_ENCODED}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


# ============================================================================
# Custom Vanna Class with Groq Backend
# ============================================================================

class VannaGroq(ChromaDB_VectorStore, OpenAI_Chat):
    """Vanna instance using Groq's OpenAI-compatible API and ChromaDB for training."""
    
    def __init__(self, openai_client=None, model=None, chroma_config=None):
        # Initialize ChromaDB with its own config (no openai client)
        ChromaDB_VectorStore.__init__(self, config=chroma_config or {})
        # Initialize OpenAI with client
        OpenAI_Chat.__init__(self, config={
            "client": openai_client,
            "model": model,
        })
        # Explicitly set client attribute (required by some Vanna methods)
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
            chroma_config={"path": "/app/chroma_data"}
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
        
        # ====================================================================
        # REAL DRONE LOGIC TRAINING (Golden Pairs)
        # ====================================================================
        sample_queries = [
            # 1. Handling the "Digital" vs "Analog" Logic (Requires Joining camera_specs)
            {
                "question": "Show me all digital nano cameras",
                "sql": """SELECT t1.title, t1.brand, t1.weight_g 
                          FROM "Products" t1 
                          JOIN camera_specs t2 ON t1.product_id = t2.product_id 
                          WHERE t1.category = 'camera' 
                          AND t1.sub_category = 'nano' 
                          AND t2.system = 'digital'"""
            },
            {
                "question": "Show me analog cameras for micro drones",
                "sql": """SELECT t1.title, t1.brand, t1.weight_g 
                          FROM "Products" t1 
                          JOIN camera_specs t2 ON t1.product_id = t2.product_id 
                          WHERE t1.category = 'camera' 
                          AND t2.size_class = 'micro' 
                          AND t2.system = 'analog'"""
            },
            {
                "question": "What digital cameras work for a 5 inch drone",
                "sql": """SELECT t1.title, t1.brand, t1.weight_g, t2.digital_platform 
                          FROM "Products" t1 
                          JOIN camera_specs t2 ON t1.product_id = t2.product_id 
                          WHERE t2.system = 'digital' 
                          AND t2.size_class IN ('micro', 'standard')"""
            },
            # 2. Handling Ecosystem Specifics (DJI O3/Walksnail/HDZero)
            {
                "question": "I need a camera compatible with DJI O3",
                "sql": """SELECT t1.title, t1.brand, t1.weight_g 
                          FROM "Products" t1 
                          JOIN camera_specs t2 ON t1.product_id = t2.product_id 
                          WHERE t2.digital_platform = 'dji'"""
            },
            {
                "question": "Show me Walksnail cameras and VTX",
                "sql": """SELECT title, brand, category, weight_g 
                          FROM "Products" 
                          WHERE brand = 'Walksnail' 
                          AND category IN ('camera', 'vtx')"""
            },
            {
                "question": "What HDZero options do you have",
                "sql": """SELECT title, brand, category, weight_g 
                          FROM "Products" 
                          WHERE brand = 'HDZero' 
                          ORDER BY category"""
            },
            # 3. Handling VTX Power (Requires Joining vtx_specs)
            {
                "question": "Find me a high power VTX over 800mW",
                "sql": """SELECT t1.title, t1.brand, t2.power_mw_max 
                          FROM "Products" t1 
                          JOIN vtx_specs t2 ON t1.product_id = t2.product_id 
                          WHERE t2.power_mw_max >= 800"""
            },
            {
                "question": "Show me analog VTX options",
                "sql": """SELECT t1.title, t1.brand, t1.weight_g, t2.power_mw_max 
                          FROM "Products" t1 
                          JOIN vtx_specs t2 ON t1.product_id = t2.product_id 
                          WHERE t2.system = 'analog'"""
            },
            # 4. Handling Receiver Protocols (ELRS/Crossfire)
            {
                "question": "Do you have any ELRS receivers?",
                "sql": """SELECT title, brand, weight_g 
                          FROM "Products" 
                          WHERE category = 'receiver' 
                          AND LOWER(sub_category) = 'elrs'"""
            },
            {
                "question": "Show me Crossfire receivers",
                "sql": """SELECT title, brand, weight_g 
                          FROM "Products" 
                          WHERE category = 'receiver' 
                          AND LOWER(sub_category) = 'crsf'"""
            },
            # 5. Handling Antenna Connectors (SMA/UFL/MMCX)
            {
                "question": "I need an antenna with an SMA connector",
                "sql": """SELECT title, brand, weight_g 
                          FROM "Products" 
                          WHERE category = 'antenna' 
                          AND LOWER(sub_category) = 'sma'"""
            },
            {
                "question": "Show me UFL antennas for whoop drones",
                "sql": """SELECT title, brand, weight_g 
                          FROM "Products" 
                          WHERE category = 'antenna' 
                          AND LOWER(sub_category) = 'ufl'"""
            },
            # 6. Handling Motor searches
            {
                "question": "Show me motors for whoop drones",
                "sql": """SELECT title, brand, weight_g, sub_category 
                          FROM "Products" 
                          WHERE category = 'motor' 
                          AND sub_category = 'whoop'"""
            },
            {
                "question": "What 5 inch motors do you have",
                "sql": """SELECT title, brand, weight_g 
                          FROM "Products" 
                          WHERE category = 'motor' 
                          AND sub_category = '5inch'"""
            },
            # 7. Handling Brand searches
            {
                "question": "What Foxeer cameras do you have?",
                "sql": """SELECT title, sub_category, weight_g 
                          FROM "Products" 
                          WHERE category = 'camera' 
                          AND brand = 'Foxeer'"""
            },
            {
                "question": "Show me all Caddx products",
                "sql": """SELECT title, category, sub_category, weight_g 
                          FROM "Products" 
                          WHERE brand = 'Caddx' 
                          ORDER BY category"""
            },
            {
                "question": "What TBS products do you have",
                "sql": """SELECT title, category, weight_g 
                          FROM "Products" 
                          WHERE brand = 'TBS' 
                          ORDER BY category"""
            },
            # 8. Frame and Size queries
            {
                "question": "Show me 5 inch frames",
                "sql": """SELECT title, brand, weight_g 
                          FROM "Products" 
                          WHERE category = 'frame' 
                          AND sub_category = '5-inch'"""
            },
            {
                "question": "What whoop frames do you have",
                "sql": """SELECT title, brand, weight_g 
                          FROM "Products" 
                          WHERE category = 'frame' 
                          AND sub_category IN ('whoop', 'cinewhoop')"""
            },
            # 9. Multi-category drone build queries
            {
                "question": "Show me all components for a drone build",
                "sql": """SELECT title, category, sub_category, brand, weight_g 
                          FROM "Products" 
                          WHERE category IN ('frame', 'motor', 'vtx', 'camera', 'receiver', 'antenna', 'stack') 
                          ORDER BY category"""
            },
            {
                "question": "I want to build a 5 inch drone show me frames motors cameras and VTX",
                "sql": """SELECT title, category, brand, weight_g 
                          FROM "Products" 
                          WHERE category IN ('frame', 'motor', 'camera', 'vtx') 
                          ORDER BY category, CAST(weight_g AS INTEGER)"""
            }
        ]
        
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
