"""Serve the viewer locally:  python serve.py [port]  ->  http://localhost:8000"""
import functools
import http.server
import sys
from pathlib import Path

port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(Path(__file__).parent / "viewer"))
print(f"SLAM Lab viewer on http://localhost:{port}")
http.server.ThreadingHTTPServer(("", port), handler).serve_forever()
