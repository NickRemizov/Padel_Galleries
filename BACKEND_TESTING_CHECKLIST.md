# Backend API Testing Checklist

## Status: Testing PostgreSQL Integration

### ✅ Already Tested (Working)

1. **GET /api/v2/config** - Get recognition configuration
   - Status: ✅ Working
   - Returns: confidence thresholds, quality filters, training params
   - Database: PostgreSQL

2. **GET /api/v2/statistics** - Get training statistics  
   - Status: ✅ Working
   - Returns: `{"people_count":104,"total_faces":1138,"unique_photos":942}`
   - Database: PostgreSQL

### 🔄 To Test Now

#### Training Endpoints

3. **GET /api/v2/train/history** - Get training session history
   \`\`\`bash
   curl -s http://localhost:8001/api/v2/train/history
   \`\`\`
   Expected: List of training sessions from `training_sessions` table

4. **GET /api/v2/train/status/{session_id}** - Get training session status
   \`\`\`bash
   # First get a session_id from history, then:
   curl -s http://localhost:8001/api/v2/train/status/SESSION_ID
   \`\`\`
   Expected: Status of specific training session

5. **POST /api/v2/train/prepare** - Prepare training session
   \`\`\`bash
   curl -X POST http://localhost:8001/api/v2/train/prepare \
     -H "Content-Type: application/json" \
     -d '{"force_retrain": false}'
   \`\`\`
   Expected: Creates new training session, returns session_id

6. **POST /api/v2/train/execute** - Execute training
   \`\`\`bash
   curl -X POST http://localhost:8001/api/v2/train/execute \
     -H "Content-Type: application/json" \
     -d '{"session_id": "SESSION_ID"}'
   \`\`\`
   Expected: Starts training process

#### Recognition Endpoints

7. **POST /api/v2/recognize/batch** - Batch face recognition
   \`\`\`bash
   curl -X POST http://localhost:8001/api/v2/recognize/batch \
     -H "Content-Type: application/json" \
     -d '{
       "photos": [],
       "settings": {
         "confidence_threshold": 0.6,
         "use_context": true,
         "max_results": 5
       }
     }'
   \`\`\`
   Expected: Empty results for empty photos array

8. **POST /api/v2/detect-faces** - Detect faces in image
   - Requires: multipart/form-data with image file
   - Test from frontend or with curl file upload

9. **POST /api/v2/recognize-face** - Recognize single face
   - Requires: face embedding data
   - Test from frontend

### 📋 Testing Commands (Run on Server)

\`\`\`bash
# Copy script to server
cd /home/nickr/python

# Make executable
chmod +x test_backend_endpoints.sh

# Run tests
bash test_backend_endpoints.sh

# Or test manually:
curl -s http://localhost:8001/api/v2/train/history
curl -s http://localhost:8001/api/v2/statistics
\`\`\`

### 🎯 Success Criteria

- ✅ All GET endpoints return data without errors
- ✅ No "supabase" errors in logs
- ✅ PostgreSQL queries execute successfully
- ✅ Response times < 1 second for simple queries

### 🐛 Troubleshooting

If endpoints return 500 errors:
\`\`\`bash
# Check logs for errors
tail -50 /home/nickr/python/fastapi.log | grep -i error

# Check PostgreSQL connection
psql $POSTGRES_URL -c "SELECT COUNT(*) FROM verified_faces;"
\`\`\`

If endpoints return 404:
\`\`\`bash
# Check available endpoints
curl -s http://localhost:8001/openapi.json | grep -o '"/api/v2/[^"]*"' | sort -u
