#!/bin/bash
# Start annotation tool for Nakhla

cd "$(dirname "$0")"

echo "======================================"
echo "  Starting Nakhla's Annotation Tool"
echo "======================================"
echo ""
echo "Opening in browser..."
echo "URL: http://localhost:8001/nakhla_annotate.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Python HTTP server and open browser
python3 -m http.server 8001 &
SERVER_PID=$!

# Wait a moment for server to start
sleep 2

# Open in browser
open "http://localhost:8001/nakhla_annotate.html"

# Wait for server
wait $SERVER_PID
