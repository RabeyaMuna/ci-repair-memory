#!/bin/bash
# Start annotation tool for Rabeya

cd "$(dirname "$0")"

echo "======================================"
echo "  Starting Rabeya's Annotation Tool"
echo "======================================"
echo ""
echo "Opening in browser..."
echo "URL: http://localhost:8002/rabeya_annotate.html"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start Python HTTP server and open browser
python3 -m http.server 8002 &
SERVER_PID=$!

# Wait a moment for server to start
sleep 2

# Open in browser
open "http://localhost:8002/rabeya_annotate.html"

# Wait for server
wait $SERVER_PID
