#!/bin/bash
# Start annotation tool for Rabeya

cd "$(dirname "$0")"

PORT=8002

# Function to check if port is in use
check_port() {
    lsof -i:$1 >/dev/null 2>&1
}

# Kill any existing process on the port
if check_port $PORT; then
    echo "⚠️  Port $PORT is in use. Cleaning up..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null
    sleep 1
fi

# Find an available port if default is still busy
while check_port $PORT && [ $PORT -lt 8020 ]; do
    echo "Port $PORT is busy, trying port $((PORT + 1))..."
    PORT=$((PORT + 1))
done

echo "======================================"
echo "  Starting Rabeya's Annotation Tool"
echo "======================================"
echo ""
echo "Opening in browser..."
echo "URL: http://localhost:$PORT/rabeya_annotate.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Python HTTP server and open browser
python3 -m http.server $PORT &
SERVER_PID=$!

# Wait a moment for server to start
sleep 2

# Open in browser
open "http://localhost:$PORT/rabeya_annotate.html"

# Wait for server
wait $SERVER_PID
