#!/bin/bash

# Backend API Testing Script
# Run on server: bash test_backend_endpoints.sh

BASE_URL="http://localhost:8001"
API_PREFIX="/api/v2"

echo "======================================"
echo "BACKEND API TESTING - PostgreSQL"
echo "======================================"
echo ""

# Test 1: Config endpoints
echo "1. GET /config - Get recognition config"
curl -s "${BASE_URL}${API_PREFIX}/config"
echo -e "\n"

# Test 2: Statistics
echo "2. GET /statistics - Get training statistics"
curl -s "${BASE_URL}${API_PREFIX}/statistics"
echo -e "\n"

# Test 3: Training history
echo "3. GET /train/history - Get training history"
curl -s "${BASE_URL}${API_PREFIX}/train/history"
echo -e "\n"

# Test 4: Training prepare (requires data)
echo "4. POST /train/prepare - Prepare training session"
echo "Skipped - requires training data"
echo -e "\n"

# Test 5: Recognition - detect faces (needs image)
echo "5. POST /detect-faces - Detect faces in image"
echo "Skipped - requires image upload"
echo -e "\n"

# Test 6: Batch recognize (needs config)
echo "6. POST /recognize/batch - Batch recognition"
curl -s -X POST "${BASE_URL}${API_PREFIX}/recognize/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "photos": [],
    "settings": {
      "confidence_threshold": 0.6,
      "use_context": true,
      "max_results": 5
    }
  }'
echo -e "\n"

# Test 7: Check database connection
echo "7. Database Connection Check"
echo "Checking if PostgreSQL client is working..."
curl -s "${BASE_URL}${API_PREFIX}/statistics" | grep -q "people_count" && echo "✅ PostgreSQL connection OK" || echo "❌ PostgreSQL connection FAILED"
echo -e "\n"

# Test 8: Check for Supabase errors in logs
echo "8. Checking logs for errors"
tail -50 /home/nickr/python/fastapi.log | grep -i "error\|exception\|traceback" && echo "⚠️ Errors found in logs" || echo "✅ No errors in recent logs"
echo -e "\n"

echo "======================================"
echo "TESTING COMPLETE"
echo "======================================"
