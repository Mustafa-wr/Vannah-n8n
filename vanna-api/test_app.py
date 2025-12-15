"""
Unit tests for Vanna API endpoints.
Uses pytest with FastAPI TestClient for API testing.
"""

import os
import pytest
from unittest.mock import Mock, patch, MagicMock
from fastapi.testclient import TestClient
import pandas as pd

# Set test environment variables before importing app
os.environ["GROQ_API_KEY"] = "test-key"
os.environ["GROQ_MODEL"] = "llama-3.3-70b-versatile"
os.environ["INTERNAL_API_KEY"] = "test-internal-key"
os.environ["DB_HOST"] = "localhost"
os.environ["DB_PORT"] = "5432"
os.environ["DB_NAME"] = "testdb"
os.environ["DB_USER"] = "testuser"
os.environ["DB_PASS"] = "testpass"


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def client():
    """Create a test client with mocked Vanna instance."""
    with patch("app.get_vanna") as mock_get_vanna:
        # Create mock Vanna instance
        mock_vn = Mock()
        mock_get_vanna.return_value = mock_vn
        
        from app import app
        with TestClient(app) as test_client:
            yield test_client, mock_vn


@pytest.fixture
def auth_headers():
    """Return valid authentication headers."""
    return {"x-api-key": "test-internal-key"}


@pytest.fixture
def invalid_auth_headers():
    """Return invalid authentication headers."""
    return {"x-api-key": "wrong-key"}


# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_check_returns_200(self, client):
        """Health endpoint should return 200 and status healthy."""
        test_client, _ = client
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
    
    def test_health_check_no_auth_required(self, client):
        """Health endpoint should not require authentication."""
        test_client, _ = client
        response = test_client.get("/health")
        assert response.status_code == 200


# ============================================================================
# Authentication Tests
# ============================================================================

class TestAuthentication:
    """Tests for API key authentication."""
    
    def test_ask_requires_api_key(self, client):
        """POST /ask should require x-api-key header."""
        test_client, _ = client
        response = test_client.post(
            "/ask",
            json={"question": "test question"}
        )
        assert response.status_code == 401
    
    def test_ask_rejects_invalid_api_key(self, client, invalid_auth_headers):
        """POST /ask should reject invalid API key."""
        test_client, _ = client
        response = test_client.post(
            "/ask",
            json={"question": "test question"},
            headers=invalid_auth_headers
        )
        assert response.status_code == 401
    
    def test_train_requires_api_key(self, client):
        """POST /train should require x-api-key header."""
        test_client, _ = client
        response = test_client.post("/train")
        assert response.status_code == 401
    
    def test_tables_requires_api_key(self, client):
        """GET /tables should require x-api-key header."""
        test_client, _ = client
        response = test_client.get("/tables")
        assert response.status_code == 401


# ============================================================================
# Ask Endpoint Tests
# ============================================================================

class TestAskEndpoint:
    """Tests for POST /ask endpoint."""
    
    def test_ask_generates_sql(self, client, auth_headers):
        """Should generate SQL from natural language question."""
        test_client, mock_vn = client
        
        # Mock Vanna to return SQL
        mock_vn.generate_sql.return_value = "SELECT * FROM products WHERE price < 500"
        
        # Mock run_sql
        with patch("app.run_sql") as mock_run_sql:
            mock_run_sql.return_value = pd.DataFrame([
                {"id": 1, "name": "Drone A", "price": 399},
                {"id": 2, "name": "Drone B", "price": 450}
            ])
            
            response = test_client.post(
                "/ask",
                json={"question": "Show me drones under $500"},
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["question"] == "Show me drones under $500"
        assert data["sql"] == "SELECT * FROM products WHERE price < 500"
        assert data["data"] is not None
        assert len(data["data"]) == 2
    
    def test_ask_without_execute(self, client, auth_headers):
        """Should return SQL without executing when execute=False."""
        test_client, mock_vn = client
        
        mock_vn.generate_sql.return_value = "SELECT * FROM products"
        
        response = test_client.post(
            "/ask",
            json={"question": "Show all products", "execute": False},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["sql"] == "SELECT * FROM products"
        assert data["data"] is None  # Not executed
    
    def test_ask_handles_sql_generation_failure(self, client, auth_headers):
        """Should handle case when SQL cannot be generated."""
        test_client, mock_vn = client
        
        mock_vn.generate_sql.return_value = None
        
        response = test_client.post(
            "/ask",
            json={"question": "Invalid question"},
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["sql"] is None
        assert "Could not generate SQL" in data["error"]
    
    def test_ask_handles_sql_execution_failure(self, client, auth_headers):
        """Should return SQL even if execution fails."""
        test_client, mock_vn = client
        
        mock_vn.generate_sql.return_value = "SELECT * FROM nonexistent_table"
        
        with patch("app.run_sql") as mock_run_sql:
            mock_run_sql.side_effect = Exception("Table does not exist")
            
            response = test_client.post(
                "/ask",
                json={"question": "Show data from bad table"},
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["sql"] == "SELECT * FROM nonexistent_table"
        assert "SQL execution failed" in data["error"]
    
    def test_ask_cleans_sql_markdown(self, client, auth_headers):
        """Should clean markdown formatting from generated SQL."""
        test_client, mock_vn = client
        
        # Simulate LLM returning SQL wrapped in markdown
        mock_vn.generate_sql.return_value = "```sql\nSELECT * FROM products\n```"
        
        with patch("app.run_sql") as mock_run_sql:
            mock_run_sql.return_value = pd.DataFrame([{"id": 1}])
            
            response = test_client.post(
                "/ask",
                json={"question": "Show products"},
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        # SQL should be cleaned
        assert "```" not in data["sql"]


# ============================================================================
# Train Endpoint Tests
# ============================================================================

class TestTrainEndpoint:
    """Tests for POST /train endpoint."""
    
    def test_train_discovers_tables(self, client, auth_headers):
        """Should discover and train on database tables."""
        test_client, mock_vn = client
        
        with patch("app.create_engine") as mock_engine:
            with patch("app.inspect") as mock_inspect:
                # Mock inspector
                mock_inspector = Mock()
                mock_inspect.return_value = mock_inspector
                mock_inspector.get_table_names.return_value = ["products", "orders"]
                mock_inspector.get_columns.return_value = [
                    {"name": "id", "type": "INTEGER", "nullable": False},
                    {"name": "name", "type": "VARCHAR", "nullable": True}
                ]
                mock_inspector.get_pk_constraint.return_value = {"constrained_columns": ["id"]}
                mock_inspector.get_foreign_keys.return_value = []
                
                response = test_client.post("/train", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "Training completed" in data["message"]
        assert "products" in data["tables_trained"]
        assert "orders" in data["tables_trained"]
    
    def test_train_handles_empty_database(self, client, auth_headers):
        """Should handle database with no tables."""
        test_client, mock_vn = client
        
        with patch("app.create_engine") as mock_engine:
            with patch("app.inspect") as mock_inspect:
                mock_inspector = Mock()
                mock_inspect.return_value = mock_inspector
                mock_inspector.get_table_names.return_value = []
                
                response = test_client.post("/train", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["tables_trained"] == []


# ============================================================================
# Tables Endpoint Tests
# ============================================================================

class TestTablesEndpoint:
    """Tests for GET /tables endpoint."""
    
    def test_tables_lists_all_tables(self, client, auth_headers):
        """Should list all tables in the database."""
        test_client, _ = client
        
        with patch("app.create_engine") as mock_engine:
            with patch("app.inspect") as mock_inspect:
                mock_inspector = Mock()
                mock_inspect.return_value = mock_inspector
                mock_inspector.get_table_names.return_value = ["products", "orders", "users"]
                
                response = test_client.get("/tables", headers=auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert "tables" in data
        assert len(data["tables"]) == 3


# ============================================================================
# Integration-style Tests (with mocked DB)
# ============================================================================

class TestComplexQueries:
    """Tests for complex natural language queries."""
    
    def test_complex_filter_query(self, client, auth_headers):
        """Should handle complex filter queries like 'drones under $500 with weight < 250g'."""
        test_client, mock_vn = client
        
        expected_sql = """
            SELECT * FROM products 
            WHERE price < 500 AND weight < 250 
            ORDER BY price ASC
        """.strip()
        
        mock_vn.generate_sql.return_value = expected_sql
        
        with patch("app.run_sql") as mock_run_sql:
            mock_run_sql.return_value = pd.DataFrame([
                {"id": 1, "name": "Mini Drone", "price": 299, "weight": 200}
            ])
            
            response = test_client.post(
                "/ask",
                json={"question": "Show me drones under $500 with weight less than 250g"},
                headers=auth_headers
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["data"] is not None
        assert len(data["data"]) == 1


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
