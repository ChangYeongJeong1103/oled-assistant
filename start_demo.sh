#!/bin/bash
# Quick Demo Script with ngrok tunneling
# Usage: ./start_demo.sh
echo "================================================="
echo "🌍 Starting Global Demo via ngrok"
echo "================================================="

# Check ngrok
if ! command -v ngrok &> /dev/null; then
    echo "❌ ngrok not found!"
    echo "   Please install: brew install ngrok/ngrok/ngrok"
    echo "   And authenticate: ngrok config add-authtoken <YOUR_TOKEN>"
    exit 1
fi

# Kill existing ngrok processes
pkill ngrok

# Create logs directory if it doesn't exist
mkdir -p logs

# Start App in Background
source oled/bin/activate
cd src
streamlit run app.py --server.port 8501 --server.headless true > ../logs/app_output.log 2>&1 &
APP_PID=$!
echo "✅ Streamlit app started (PID: $APP_PID)"

# Start Tunnel
echo "🚇 Starting ngrok tunnel..."
ngrok http 8501 > /dev/null &
NGROK_PID=$!

sleep 5

# Get URL
PUBLIC_URL=$(curl -s http://localhost:4040/api/tunnels | grep -o "https://[a-zA-Z0-9-]*\.ngrok-free\.app")

echo "================================================="
echo "✅ DEMO IS LIVE!"
echo "🔗 Share this URL with your colleagues:"
echo ""
echo "   $PUBLIC_URL"
echo ""
echo "================================================="
echo "Press CTRL+C to stop the demo."

# Wait
wait $APP_PID
