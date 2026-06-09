# Copyright (c) 2025 Apple Inc. Licensed under MIT License.

import pandas as pd
from fastapi.testclient import TestClient
from embedding_atlas.server import is_dangerous_query, make_server
from embedding_atlas.data_source import DataSource


class TestIsDangerousQuery:
    """Verify that is_dangerous_query properly classifies dangerous vs safe statements."""

    def test_safe_read_queries(self):
        assert is_dangerous_query("SELECT * FROM dataset") is False
        assert is_dangerous_query("WITH q AS (SELECT 1) SELECT * FROM q") is False
        assert is_dangerous_query("DESCRIBE dataset") is False
        assert is_dangerous_query("EXPLAIN SELECT * FROM dataset") is False
        assert is_dangerous_query("SHOW tables") is False
        assert is_dangerous_query("PRAGMA table_info('dataset')") is False

    def test_safe_mosaic_write_queries(self):
        """These DDL/DML queries are legitimately used by the Mosaic frontend."""
        assert is_dangerous_query("ALTER TABLE dataset ADD COLUMN IF NOT EXISTS col INTEGER DEFAULT 0") is False
        assert is_dangerous_query("UPDATE dataset SET col = 1") is False
        assert is_dangerous_query("CREATE TABLE t AS SELECT * FROM dataset") is False
        assert is_dangerous_query("INSERT INTO dataset VALUES (1)") is False

    def test_dangerous_queries(self):
        assert is_dangerous_query("DROP TABLE dataset") is True
        assert is_dangerous_query("DELETE FROM dataset") is True
        assert is_dangerous_query("TRUNCATE TABLE dataset") is True
        assert is_dangerous_query("INSTALL httpfs") is True
        assert is_dangerous_query("LOAD httpfs") is True
        assert is_dangerous_query("ATTACH 'other.db'") is True
        assert is_dangerous_query("DETACH other") is True

    def test_multi_statement_with_dangerous(self):
        assert is_dangerous_query("SELECT * FROM dataset; DROP TABLE dataset") is True

    def test_string_literals_not_flagged(self):
        """Keywords inside string literals should not trigger false positives."""
        assert is_dangerous_query("SELECT * FROM dataset WHERE col = 'drop table'") is False
        assert is_dangerous_query("SELECT * FROM dataset WHERE col = \"delete from\"") is False
        assert is_dangerous_query("SELECT * FROM dataset WHERE col = $$drop table$$") is False

    def test_comment_bypass_blocked(self):
        """Keywords hidden in comments should still be caught."""
        assert is_dangerous_query("-- safe comment\nDROP TABLE dataset") is True
        assert is_dangerous_query("/* safe */ DROP TABLE dataset") is True


class TestServerQueryEndpoint:
    """Verify that the /data/query endpoint enforces the dangerous query check."""

    def setup_method(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        data_source = DataSource("test_id", df, {})
        app = make_server(data_source, static_path=".", duckdb_uri="server")
        self.client = TestClient(app)

    def test_read_query_succeeds(self):
        payload = {"type": "json", "sql": "SELECT * FROM dataset WHERE a = 1"}
        response = self.client.post("/data/query", json=payload)
        assert response.status_code == 200
        assert response.json() == [{"a": 1, "b": "x"}]

    def test_dangerous_query_rejected(self):
        payload = {"type": "exec", "sql": "DROP TABLE dataset"}
        response = self.client.post("/data/query", json=payload)
        assert response.status_code == 400
        assert "Dangerous SQL operations are not allowed" in response.json()["error"]

    def test_safe_alter_table_succeeds(self):
        """ALTER TABLE is used by Mosaic and should be allowed."""
        payload = {"type": "exec", "sql": "ALTER TABLE dataset ADD COLUMN IF NOT EXISTS test_col INTEGER DEFAULT 0"}
        response = self.client.post("/data/query", json=payload)
        assert response.status_code == 200
