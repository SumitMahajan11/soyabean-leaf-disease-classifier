"""
Simple HTTP server to serve the soybean disease detection frontend
"""
import http.server
import socketserver
import os
from pathlib import Path

# Change to the frontend directory
frontend_dir = Path(__file__).parent / "frontend"

class FrontendHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(frontend_dir), **kwargs)

def run_server(port=8000):
    """Run the frontend server"""
    print(f"Serving soybean disease detection frontend at http://localhost:{port}")
    print(f"Serving from directory: {frontend_dir}")
    
    # Change to the frontend directory
    os.chdir(frontend_dir)
    
    with socketserver.TCPServer(("", port), FrontendHTTPRequestHandler) as httpd:
        print(f"Server started. Access the frontend at: http://localhost:{port}")
        print("Press Ctrl+C to stop the server")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")
            return

if __name__ == "__main__":
    run_server()