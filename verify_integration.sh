#!/bin/bash

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "================================"
echo "Integration Verification Check"
echo "================================"
echo ""

# Check Backend Files
echo -e "${YELLOW}Checking Backend Files...${NC}"
if [ -f "api_server.py" ]; then
    echo -e "${GREEN}✓${NC} api_server.py exists"
else
    echo -e "${RED}✗${NC} api_server.py missing"
fi

if [ -f "requirements-api.txt" ]; then
    echo -e "${GREEN}✓${NC} requirements-api.txt exists"
else
    echo -e "${RED}✗${NC} requirements-api.txt missing"
fi

if [ -f ".env.example" ]; then
    echo -e "${GREEN}✓${NC} .env.example exists"
else
    echo -e "${RED}✗${NC} .env.example missing"
fi

# Check Frontend Files
echo ""
echo -e "${YELLOW}Checking Frontend Files...${NC}"
if [ -f "face-attend-main/src/lib/face-api.ts" ]; then
    echo -e "${GREEN}✓${NC} face-api.ts exists"
else
    echo -e "${RED}✗${NC} face-api.ts missing"
fi

if [ -f "face-attend-main/.env.example" ]; then
    echo -e "${GREEN}✓${NC} frontend .env.example exists"
else
    echo -e "${RED}✗${NC} frontend .env.example missing"
fi

if grep -q "import.*face-api" "face-attend-main/src/pages/FaceRegister.tsx" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} FaceRegister.tsx uses face-api"
else
    echo -e "${RED}✗${NC} FaceRegister.tsx not updated"
fi

if grep -q "import.*face-api" "face-attend-main/src/pages/MarkAttendance.tsx" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} MarkAttendance.tsx uses face-api"
else
    echo -e "${RED}✗${NC} MarkAttendance.tsx not updated"
fi

# Check Documentation
echo ""
echo -e "${YELLOW}Checking Documentation...${NC}"
if [ -f "FULLSTACK_SETUP_GUIDE.md" ]; then
    echo -e "${GREEN}✓${NC} FULLSTACK_SETUP_GUIDE.md exists"
else
    echo -e "${RED}✗${NC} FULLSTACK_SETUP_GUIDE.md missing"
fi

if [ -f "INTEGRATION_SUMMARY.md" ]; then
    echo -e "${GREEN}✓${NC} INTEGRATION_SUMMARY.md exists"
else
    echo -e "${RED}✗${NC} INTEGRATION_SUMMARY.md missing"
fi

# Python Syntax Check
echo ""
echo -e "${YELLOW}Checking Python Syntax...${NC}"
if python3 -m py_compile api_server.py 2>/dev/null; then
    echo -e "${GREEN}✓${NC} api_server.py is valid Python"
else
    echo -e "${RED}✗${NC} api_server.py has syntax errors"
fi

# Check MongoDB connectivity
echo ""
echo -e "${YELLOW}Checking Dependencies...${NC}"
if python3 -c "import flask" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Flask is installed"
else
    echo -e "${YELLOW}⚠${NC} Flask not installed - run: pip install -r requirements-api.txt"
fi

if python3 -c "import pymongo" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} pymongo is installed"
else
    echo -e "${YELLOW}⚠${NC} pymongo not installed - run: pip install -r requirements-api.txt"
fi

if python3 -c "import tensorflow" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} TensorFlow is installed"
else
    echo -e "${YELLOW}⚠${NC} TensorFlow not installed"
fi

# Check Node/npm
echo ""
echo -e "${YELLOW}Checking Frontend Environment...${NC}"
if command -v node &> /dev/null; then
    NODE_VERSION=$(node -v)
    echo -e "${GREEN}✓${NC} Node.js is installed: $NODE_VERSION"
else
    echo -e "${RED}✗${NC} Node.js not found"
fi

if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm -v)
    echo -e "${GREEN}✓${NC} npm is installed: $NPM_VERSION"
else
    echo -e "${RED}✗${NC} npm not found"
fi

# Summary
echo ""
echo "================================"
echo "Verification Complete!"
echo "================================"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Backend: pip install -r requirements-api.txt"
echo "2. Frontend: cd face-attend-main && npm install"
echo "3. Configure: cp .env.example .env"
echo "4. Start MongoDB: docker run -d -p 27017:27017 mongo:7"
echo "5. Run Backend: python api_server.py"
echo "6. Run Frontend: cd face-attend-main && npm run dev"
echo ""
echo -e "${YELLOW}For detailed setup, see:${NC} FULLSTACK_SETUP_GUIDE.md"
