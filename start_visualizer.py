#!/usr/bin/env python3
"""
Simple HTTP server to serve the web-based visualizer for TheSimulation.
Run this script to start the web server, then open http://localhost:8080 in your browser.
"""

import http.server
import socketserver
import os
import sys
import webbrowser
import threading
import time

PORT = 8080

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.dirname(os.path.abspath(__file__)), **kwargs)
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()
    
    def do_GET(self):
        # Serve the main visualizer page as default
        if self.path == '/' or self.path == '':
            self.path = '/web_visualizer.html'
        super().do_GET()

def open_browser():
    """Open the browser after a short delay to ensure server is ready"""
    time.sleep(1.5)
    webbrowser.open(f'http://localhost:{PORT}')

def main():
    print(f"TheSimulation Web Visualizer")
    print(f"============================")
    print(f"Starting HTTP server on port {PORT}")
    print(f"Serving files from: {os.path.dirname(os.path.abspath(__file__))}")
    print(f"")
    print(f"To view the visualizer:")
    print(f"1. Make sure the simulation is running (python main_async.py)")
    print(f"2. Open your browser to: http://localhost:{PORT}")
    print(f"")
    print(f"Press Ctrl+C to stop the server")
    print(f"")
    
    # Check if simulation files exist
    required_files = ['web_visualizer.html', 'simulation_viz.js']
    for file in required_files:
        if not os.path.exists(file):
            print(f"ERROR: Required file '{file}' not found!")
            print(f"Make sure you're running this from the TheSimulation directory.")
            sys.exit(1)
    
    try:
        # Start browser in a separate thread
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Start HTTP server
        with socketserver.TCPServer(("", PORT), Handler) as httpd:
            print(f"Server started successfully at http://localhost:{PORT}")
            print(f"Opening browser automatically...")
            httpd.serve_forever()
            
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except OSError as e:
        if e.errno == 48:  # Address already in use
            print(f"ERROR: Port {PORT} is already in use!")
            print(f"Either:")
            print(f"  1. Stop the process using port {PORT}")
            print(f"  2. Change the PORT variable in this script")
        else:
            print(f"ERROR: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()