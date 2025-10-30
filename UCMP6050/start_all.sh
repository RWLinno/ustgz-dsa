#!/bin/bash

# CTIS Complete Startup Script
# Starts both API server and Web demo

echo "🚀 CTIS Complete System Startup"
echo "================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install minimal dependencies if needed
if ! python -c "import flask" 2>/dev/null; then
    echo -e "${YELLOW}Installing minimal dependencies...${NC}"
    pip install -q flask flask-cors torch numpy pandas
fi

# Generate demo data if not exists
if [ ! -f "demo/data/pickup_points.json" ]; then
    echo -e "${YELLOW}Generating demo data...${NC}"
    python generate_demo_data.py
fi

# Kill existing processes on ports 5000 and 8000
echo -e "${YELLOW}Checking for existing processes...${NC}"
lsof -ti:5000 | xargs kill -9 2>/dev/null
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 1

# Start API server
echo -e "${YELLOW}Starting API server on port 5000...${NC}"
python src/api.py > logs/api.log 2>&1 &
API_PID=$!
sleep 3

# Check if API is running
if ps -p $API_PID > /dev/null; then
    echo -e "${GREEN}✓ API server started (PID: $API_PID)${NC}"
else
    echo -e "${RED}✗ API server failed to start${NC}"
    echo "Check logs/api.log for errors"
fi

# Start web demo
echo -e "${YELLOW}Starting web demo on port 8000...${NC}"
cd demo
python3 -m http.server 8000 > ../logs/demo.log 2>&1 &
DEMO_PID=$!
cd ..
sleep 2

if ps -p $DEMO_PID > /dev/null; then
    echo -e "${GREEN}✓ Web demo started (PID: $DEMO_PID)${NC}"
else
    echo -e "${RED}✗ Web demo failed to start${NC}"
fi

# Create logs directory
mkdir -p logs

echo ""
echo -e "${GREEN}✅ CTIS System is running!${NC}"
echo ""
echo "═══════════════════════════════════════════"
echo "  🔌 API Server:  http://localhost:5000"
echo "  🌐 Web Demo:    http://localhost:8000"
echo "═══════════════════════════════════════════"
echo ""
echo "📍 Access Points:"
echo "  - Test Page:    http://localhost:8000/test.html"
echo "  - Full Demo:    http://localhost:8000/index.html"
echo ""
echo "🔧 New Features:"
echo "  ✅ Curved route paths with animation"
echo "  ✅ Hexagon heatmap (Toggle Heatmap button)"
echo "  ✅ 6 role colors on map"
echo "  ✅ Fixed chart heights"
echo "  ✅ System menu with export/settings/help/about"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Save PIDs for cleanup
echo $API_PID > .api.pid
echo $DEMO_PID > .demo.pid

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    kill $API_PID 2>/dev/null
    kill $DEMO_PID 2>/dev/null
    rm -f .api.pid .demo.pid
    echo -e "${GREEN}✓ Services stopped${NC}"
    deactivate 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Wait for user interrupt
wait

