#!/usr/bin/env python3
"""
Server startup script that should stay running
"""
import uvicorn
import signal
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_dir))

from server import app

def signal_handler(signum, frame):
    print(f"\nReceived signal {signum}. Shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Handle shutdown signals gracefully
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print("🚀 Starting Allie server...")
    print("📡 Server will be available at: http://127.0.0.1:8001")
    print("🔄 Quick topics endpoint: http://127.0.0.1:8001/api/learning/quick-topics")
    print("📋 Facts endpoint: http://127.0.0.1:8001/api/facts")
    print("🌐 UI: http://127.0.0.1:8001/ui")
    print("\n⏹️  Press Ctrl+C to stop\n")
    
    try:
        # Run with specific config to prevent shutdown
        uvicorn.run(
            app, 
            host="127.0.0.1", 
            port=8001,
            log_level="info",
            access_log=True,
            reload=False,  # Disable auto-reload
            use_colors=True
        )
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")
        sys.exit(1)