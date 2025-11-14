"""
API Integration Tests

Tests the FastAPI endpoints with real data references from the database.

Author: Equipo ADX
Date: 2025-11-13
"""

import requests
import json
from pathlib import Path

import pandas as pd

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health endpoint."""
    response = requests.get(f"{API_BASE_URL}/health/")
    assert response.status_code == 200
    data = response.json()
    assert data['status'] == 'healthy'
    assert data['model_loaded'] == True
    print("Health check passed")

def test_recommendations_quito():
    """Test recommendations for Quito user."""
    payload = {
        "id_persona": 21096.0,
        "ciudad": "Quito",
        "hora": 14,
        "k": 5
    }
    
    response = requests.post(f"{API_BASE_URL}/recommendations/", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert 'recommendations' in data
    assert len(data['recommendations']) <= 5
    assert data['filtered_by_location'] == True
    assert data['filtered_by_time'] == True
    
    # Verify all recommendations are from Quito
    for rec in data['recommendations']:
        assert rec['ciudad'] == 'Quito', f"Expected Quito, got {rec['ciudad']}"
    
    print(f"Recommendations for Quito: {len(data['recommendations'])} items")
    for i, rec in enumerate(data['recommendations'], 1):
        print(f"  {i}. {rec['establecimiento']} (prob: {rec['probability']:.4f})")

def test_recommendations_guayaquil():
    """Test recommendations for Guayaquil user."""
    payload = {
        "id_persona": 21249.0,
        "ciudad": "Guayaquil",
        "hora": 20,
        "k": 5
    }
    
    response = requests.post(f"{API_BASE_URL}/recommendations/", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    
    # Verify all recommendations are from Guayaquil
    for rec in data['recommendations']:
        assert rec['ciudad'] == 'Guayaquil', f"Expected Guayaquil, got {rec['ciudad']}"
    
    print(f"Recommendations for Guayaquil: {len(data['recommendations'])} items")

def test_recommendations_cuenca():
    """Test recommendations for Cuenca user."""
    payload = {
        "id_persona": 21275.0,
        "ciudad": "Cuenca",
        "hora": 12,
        "k": 3
    }
    
    response = requests.post(f"{API_BASE_URL}/recommendations/", json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data['recommendations']) <= 3
    
    print(f"Recommendations for Cuenca: {len(data['recommendations'])} items")

def test_morning_recommendations():
    """Test morning recommendations (8 AM)."""
    payload = {
        "id_persona": 21096.0,
        "ciudad": "Quito",
        "hora": 8,
        "k": 5
    }
    
    response = requests.post(f"{API_BASE_URL}/recommendations/", json=payload)
    assert response.status_code == 200
    print("Morning recommendations passed")

def test_evening_recommendations():
    """Test evening recommendations (8 PM)."""
    payload = {
        "id_persona": 21096.0,
        "ciudad": "Quito",
        "hora": 20,
        "k": 5
    }
    
    response = requests.post(f"{API_BASE_URL}/recommendations/", json=payload)
    assert response.status_code == 200
    print("Evening recommendations passed")

def test_concurrent_requests():
    """Test multiple concurrent requests."""
    import concurrent.futures
    
    payloads = [
        {"id_persona": 21096.0, "ciudad": "Quito", "hora": 14, "k": 5},
        {"id_persona": 21249.0, "ciudad": "Guayaquil", "hora": 15, "k": 5},
        {"id_persona": 21275.0, "ciudad": "Cuenca", "hora": 16, "k": 5},
    ]
    
    def make_request(payload):
        response = requests.post(f"{API_BASE_URL}/recommendations/", json=payload)
        return response.status_code == 200
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(make_request, payloads))
    
    assert all(results), "Some concurrent requests failed"
    print(f"Concurrent requests passed: {len(results)} requests")

def load_sample_users():
    """Load sample users from the database for testing."""
    data_path = Path("data/01-raw/df_extendida_clean.parquet")
    
    if data_path.exists():
        df = pd.read_parquet(data_path)
        
        # Get unique users per city
        sample_users = []
        
        for ciudad in ['Quito', 'Guayaquil', 'Cuenca']:
            city_users = df[df['Ciudad'] == ciudad]['Id_Persona'].unique()
            if len(city_users) > 0:
                sample_users.append({
                    'ciudad': ciudad,
                    'id_persona': float(city_users[0])
                })
        
        return sample_users
    
    return []

def run_all_tests():
    """Run all API tests."""
    print("Starting API tests...")
    print("=" * 50)
    
    try:
        test_health_check()
        print("=" * 50)
        
        test_recommendations_quito()
        print("=" * 50)
        
        test_recommendations_guayaquil()
        print("=" * 50)
        
        test_recommendations_cuenca()
        print("=" * 50)
        
        test_morning_recommendations()
        print("=" * 50)
        
        test_evening_recommendations()
        print("=" * 50)
        
        test_concurrent_requests()
        print("=" * 50)
        
        print("\nAll tests passed!")
        
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        return False
    except requests.exceptions.ConnectionError:
        print("\nError: Cannot connect to API. Make sure the server is running:")
        print("  make run-api")
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
