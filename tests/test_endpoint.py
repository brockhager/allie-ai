import requests
import json
import time

# Wait a moment for server to start
time.sleep(3)

# Test the quick-topics endpoint
url = 'http://127.0.0.1:8000/api/learning/quick-topics'
payload = {'topics': ['artificial intelligence', 'machine learning']}

try:
    response = requests.post(url, json=payload, timeout=30)
    print(f'Status Code: {response.status_code}')
    if response.status_code == 200:
        print('SUCCESS: Endpoint is working!')
        data = response.json()
        print(f'Topics processed: {data.get("topics_processed", 0)}')
        print(f'Successful: {data.get("successful", 0)}')
        print(f'Facts learned: {data.get("total_facts_learned", 0)}')
    else:
        print(f'Response: {response.text}')
except requests.exceptions.ConnectionError:
    print('ERROR: Cannot connect to server - server may not be running')
except Exception as e:
    print(f'Error: {e}')