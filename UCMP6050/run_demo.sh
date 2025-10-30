#!/bin/bash

# CTIS Demo Runner Script
# Simplified version for quick demo startup

echo "🚀 CTIS Demo Runner"
echo "===================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Check if dependencies are installed
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}Installing dependencies...${NC}"
    pip install -q torch numpy pandas flask flask-cors
fi

# Generate demo data if not exists
if [ ! -f "demo/data/pickup_points.json" ]; then
    echo -e "${YELLOW}Generating demo data...${NC}"
    python generate_demo_data.py
fi

# Start API server in background
echo -e "${YELLOW}Starting API server...${NC}"
python src/api.py &
API_PID=$!
sleep 2

# Check if API is running
if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}✓ API server started (PID: $API_PID)${NC}"
else
    echo -e "${YELLOW}⚠ API server may not be running${NC}"
fi

# Start web demo
echo -e "${YELLOW}Starting web demo...${NC}"
cd demo
python3 -m http.server 8000 &
DEMO_PID=$!
cd ..

echo -e "\n${GREEN}✓ CTIS Demo is running!${NC}"
echo ""
echo "📍 Web Demo: http://localhost:8000"
echo "🔌 API Server: http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all services"

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $API_PID 2>/dev/null
    kill $DEMO_PID 2>/dev/null
    echo -e "${GREEN}✓ Services stopped${NC}"
    exit 0
}

trap cleanup INT TERM

# Wait for user interrupt
wait

