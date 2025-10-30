#!/bin/bash

# CTIS Quick Start Script
# This script sets up and runs the CTIS system

echo "🚀 CTIS Quick Start Script"
echo "================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Create necessary directories
echo -e "${YELLOW}Creating directories...${NC}"
mkdir -p data/lade
mkdir -p checkpoints
mkdir -p logs
mkdir -p figs/output
echo -e "${GREEN}✓ Directories created${NC}"

# Generate framework diagram
echo -e "${YELLOW}Generating visualizations...${NC}"
cd figs
python plot_framework.py
python plot_results.py
cd ..
echo -e "${GREEN}✓ Visualizations generated${NC}"

# Check if model checkpoint exists
if [ ! -f "checkpoints/best_model.pth" ]; then
    echo -e "${YELLOW}No trained model found. Training a demo model...${NC}"
    echo -e "${YELLOW}(This will take a while. You can skip this and use demo mode.)${NC}"
    read -p "Train model now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python src/train.py --config config.json --epochs 10 --batch_size 8
    else
        echo -e "${YELLOW}Skipping training. Demo will use random predictions.${NC}"
    fi
else
    echo -e "${GREEN}✓ Trained model found${NC}"
fi

# Function to start API server
start_api() {
    echo -e "${YELLOW}Starting API server...${NC}"
    export CONFIG_PATH=config.json
    export MODEL_PATH=checkpoints/best_model.pth
    python src/api.py &
    API_PID=$!
    echo -e "${GREEN}✓ API server started (PID: $API_PID)${NC}"
    echo "API available at: http://localhost:5000"
    echo "API endpoints:"
    echo "  - GET  /api/health"
    echo "  - GET  /api/system_status"
    echo "  - POST /api/predict"
    echo "  - POST /api/optimize_route"
    echo "  - POST /api/chat"
}

# Function to start web demo
start_demo() {
    echo -e "${YELLOW}Starting web demo...${NC}"
    cd demo
    python3 -m http.server 8000 &
    DEMO_PID=$!
    cd ..
    echo -e "${GREEN}✓ Web demo started (PID: $DEMO_PID)${NC}"
    echo "Demo available at: http://localhost:8000"
}

# Main menu
echo -e "\n${GREEN}Setup complete!${NC}"
echo -e "\nWhat would you like to do?"
echo "1) Start full system (API + Web Demo)"
echo "2) Start API server only"
echo "3) Start web demo only"
echo "4) Train model"
echo "5) Run tests"
echo "6) Exit"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        start_api
        sleep 2
        start_demo
        echo -e "\n${GREEN}✓ Full system is running!${NC}"
        echo -e "API: http://localhost:5000"
        echo -e "Demo: http://localhost:8000"
        echo -e "\nPress Ctrl+C to stop all services"
        wait
        ;;
    2)
        start_api
        echo -e "\nPress Ctrl+C to stop the API server"
        wait
        ;;
    3)
        start_demo
        echo -e "\nPress Ctrl+C to stop the web demo"
        wait
        ;;
    4)
        echo -e "${YELLOW}Training model...${NC}"
        python src/train.py --config config.json --epochs 100 --batch_size 32
        ;;
    5)
        echo -e "${YELLOW}Running tests...${NC}"
        pytest tests/ -v
        ;;
    6)
        echo -e "${GREEN}Goodbye!${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Shutting down services...${NC}"
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null
        echo -e "${GREEN}✓ API server stopped${NC}"
    fi
    if [ ! -z "$DEMO_PID" ]; then
        kill $DEMO_PID 2>/dev/null
        echo -e "${GREEN}✓ Web demo stopped${NC}"
    fi
    deactivate
    echo -e "${GREEN}✓ Virtual environment deactivated${NC}"
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# Set up cleanup trap
trap cleanup EXIT INT TERM

