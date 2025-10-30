"""
Test script to verify API endpoints
"""

import requests
import json

API_BASE = 'http://localhost:5000/api'

def test_health():
    """Test health endpoint"""
    try:
        response = requests.get(f'{API_BASE}/health', timeout=2)
        print(f'✓ Health check: {response.json()}')
        return True
    except Exception as e:
        print(f'✗ Health check failed: {e}')
        return False

def test_system_status():
    """Test system status endpoint"""
    try:
        response = requests.get(f'{API_BASE}/system_status', timeout=2)
        data = response.json()
        print(f'✓ System status: {data["total_nodes"]} nodes, {data["active_vehicles"]} vehicles')
        return True
    except Exception as e:
        print(f'✗ System status failed: {e}')
        return False

def test_prediction():
    """Test prediction endpoint"""
    try:
        payload = {
            'node_id': 0,
            'historical_data': [[10, 12, 15, 14, 16, 18, 20, 19, 17, 15]],
            'horizon': 12
        }
        response = requests.post(f'{API_BASE}/predict', json=payload, timeout=5)
        data = response.json()
        print(f'✓ Prediction: {len(data["predictions"])} timesteps predicted')
        return True
    except Exception as e:
        print(f'✗ Prediction failed: {e}')
        return False

def test_chat():
    """Test chat endpoint"""
    try:
        payload = {
            'query': 'What are the peak delivery hours?',
            'context': {'predictions': [10, 15, 20]},
            'role': 'analyst'
        }
        response = requests.post(f'{API_BASE}/chat', json=payload, timeout=5)
        data = response.json()
        print(f'✓ Chat: {data["answer"][:50]}...')
        return True
    except Exception as e:
        print(f'✗ Chat failed: {e}')
        return False

if __name__ == '__main__':
    print('🧪 Testing CTIS API Endpoints\n')
    print('Make sure API server is running: python src/api.py\n')
    
    results = []
    results.append(('Health Check', test_health()))
    results.append(('System Status', test_system_status()))
    results.append(('Prediction', test_prediction()))
    results.append(('Chat', test_chat()))
    
    print('\n' + '='*50)
    print('Test Results:')
    for name, result in results:
        status = '✓ PASS' if result else '✗ FAIL'
        print(f'  {name}: {status}')
    
    passed = sum(1 for _, r in results if r)
    print(f'\nTotal: {passed}/{len(results)} tests passed')

