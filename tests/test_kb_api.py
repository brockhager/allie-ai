#!/usr/bin/env python3
"""Test KB API endpoint"""
import sys
sys.path.insert(0, '.')

from backend.server import app
from fastapi.testclient import TestClient

client = TestClient(app)

print("Testing GET /api/kb endpoint...")
response = client.get("/api/kb")
print(f"Status: {response.status_code}")
print(f"Response: {response.json()}")
