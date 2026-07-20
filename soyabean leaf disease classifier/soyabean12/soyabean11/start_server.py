"""
Production Server Starter
Starts the soybean disease detection backend server
"""
import subprocess
import sys
import os

def start_server():
    """Start the Flask backend server"""
    try:
        # Change to the production backend directory
        os.chdir(os.path.join(os.path.dirname(__file__), 'production_backend'))
        
        # Start the Flask app
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    start_server()