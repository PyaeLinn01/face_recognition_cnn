#!/bin/bash
# Deploy Face Recognition API to Modal

echo "🚀 Face Recognition API - Modal Deployment"
echo "=========================================="
echo ""

# Check if Modal is installed
if ! command -v modal &> /dev/null; then
    echo "❌ Modal CLI not found. Installing..."
    pip install modal
    echo "✅ Modal installed. Please run: modal setup"
    echo "   Then run this script again."
    exit 1
fi

# Check if Modal is authenticated
if ! modal profile current &> /dev/null; then
    echo "❌ Modal not authenticated. Please run:"
    echo "   modal setup"
    exit 1
fi

echo "✅ Modal CLI ready"
echo ""

# Check if api_keys.json exists
if [ ! -f "api_keys.json" ]; then
    echo "📝 Generating API keys..."
    echo "1" | python3 generate_api_keys.py
else
    echo "✅ API keys already exist in api_keys.json"
fi

echo ""
echo "🔑 Your API Keys:"
cat api_keys.json | python3 -m json.tool | grep '"key"' | head -1
echo ""

echo "📋 Next Steps:"
echo ""
echo "1️⃣  Create Modal Secret (paste your MongoDB URI):"
echo "    modal secret create face-attendance-api-secrets \\"
echo "      FACE_API_KEY=\$(cat api_keys.json | python3 -c 'import json,sys; print(json.load(sys.stdin)[0][\"key\"])') \\"
echo "      MONGODB_CONNECTION_STRING=YOUR_MONGODB_URI"
echo ""
echo "2️⃣  Deploy to Modal:"
echo "    modal deploy get_started.py"
echo ""
echo "3️⃣  Test your deployment:"
echo "    Visit https://modal.com/apps"
echo ""
echo "💰 Ready to sell API keys!"
echo "   Each key in api_keys.json can be sold to customers"
