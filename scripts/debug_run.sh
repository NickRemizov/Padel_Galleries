#!/bin/bash

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=== STARTING DEBUG RUN ===${NC}"

# 1. Kill existing process
echo "1. Killing existing FastAPI process..."
PID=$(ps aux | grep "uvicorn main:app --host 0.0.0.0 --port 8001" | grep -v grep | awk '{print $2}')
if [ -n "$PID" ]; then
    kill -9 $PID
    echo "   Killed process $PID"
else
    echo "   No process found"
fi

# 2. Start FastAPI
echo "2. Starting FastAPI..."
cd /home/nickr/python
source venv/bin/activate
nohup python -m uvicorn main:app --host 0.0.0.0 --port 8001 > /tmp/fastapi.log 2>&1 &
NEW_PID=$!
echo "   Started with PID $NEW_PID"

# 3. Wait for startup
echo "3. Waiting 5 seconds for startup..."
sleep 5

# 4. Check if running
if ps -p $NEW_PID > /dev/null; then
    echo -e "${GREEN}   FastAPI is running!${NC}"
else
    echo -e "${RED}   FastAPI failed to start!${NC}"
    echo "   Last 20 lines of log:"
    tail -n 20 /tmp/fastapi.log
    exit 1
fi

# 5. Run tests
echo "4. Running tests..."
cd /home/nickr/scripts
./run_tests.sh

# 6. Show logs if failed
if [ $? -ne 0 ]; then
    echo -e "${RED}=== TESTS FAILED - SHOWING LOGS ===${NC}"
    echo "Last 50 lines of /tmp/fastapi.log:"
    echo "----------------------------------------"
    tail -n 50 /tmp/fastapi.log
    echo "----------------------------------------"
else
    echo -e "${GREEN}=== ALL TESTS PASSED ===${NC}"
fi
