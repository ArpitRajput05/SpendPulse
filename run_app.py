"""One-Click Launcher for SpendPulse.
Spins up both the FastAPI backend and Streamlit interactive dashboard simultaneously.
"""

import sys
import os
import time
import subprocess
import argparse
import signal
import urllib.request


def is_service_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="SpendPulse Unified Runner")
    parser.add_argument("--api-only", action="store_true", help="Launch only FastAPI backend")
    parser.add_argument("--ui-only", action="store_true", help="Launch only Streamlit dashboard")
    parser.add_argument("--port-api", type=int, default=8000, help="FastAPI port (default: 8000)")
    parser.add_argument("--port-ui", type=int, default=8501, help="Streamlit port (default: 8501)")
    args = parser.parse_args()

    python_executable = sys.executable

    processes = []

    def shutdown(signum=None, frame=None):
        print("\n[SpendPulse] Shutting down services...")
        for p in processes:
            try:
                p.terminate()
            except Exception:
                pass
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # 1. Start FastAPI Backend
    if not args.ui_only:
        print(f"[SpendPulse] Launching FastAPI backend on http://127.0.0.1:{args.port_api} ...")
        api_proc = subprocess.Popen(
            [python_executable, "-m", "uvicorn", "src.api:app", "--host", "127.0.0.1", "--port", str(args.port_api)],
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        processes.append(api_proc)

        # Wait for API to become ready
        health_url = f"http://127.0.0.1:{args.port_api}/api/health"
        print("[SpendPulse] Waiting for API to start...")
        for _ in range(15):
            if is_service_ready(health_url):
                print(f"[SpendPulse] FastAPI backend is live at http://127.0.0.1:{args.port_api}")
                print(f"[SpendPulse] Interactive Swagger Docs at http://127.0.0.1:{args.port_api}/docs")
                break
            time.sleep(0.5)

    # 2. Start Streamlit Frontend
    if not args.api_only:
        print(f"[SpendPulse] Launching Streamlit dashboard on http://localhost:{args.port_ui} ...")
        st_cmd = [
            python_executable,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(args.port_ui),
            "--server.headless",
            "false",
        ]
        ui_proc = subprocess.Popen(st_cmd)
        processes.append(ui_proc)

    print("\n" + "=" * 65)
    print("  💸 SPENDPULSE IS RUNNING")
    print(f"  • Frontend UI:    http://localhost:{args.port_ui}")
    print(f"  • Backend API:    http://127.0.0.1:{args.port_api}")
    print(f"  • API Swagger:    http://127.0.0.1:{args.port_api}/docs")
    print("  Press Ctrl+C to stop all services.")
    print("=" * 65 + "\n")

    try:
        # Keep launcher alive while child processes run
        while True:
            time.sleep(1)
            for p in processes:
                if p.poll() is not None:
                    print(f"[SpendPulse] Process {p.pid} exited with code {p.returncode}")
                    shutdown()
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
