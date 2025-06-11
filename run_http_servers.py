import asyncio
import subprocess
import sys
import time
from pathlib import Path


async def run_server(name: str, script_path: str, port: int) -> subprocess.Popen | None:
    """Run a single MCP server."""
    print(f"Starting {name} on port {port}...")

    process = subprocess.Popen(
        [sys.executable, script_path],
        cwd=Path(__file__).parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Wait a moment for server to start
    time.sleep(2)

    if process.poll() is None:
        print(f"✅ {name} started successfully on http://127.0.0.1:{port}/mcp")
    else:
        stdout, stderr = process.communicate()
        print(f"❌ {name} failed to start:")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return None

    return process


async def main():
    """Run all HTTP servers."""
    print("🚀 Starting MCP HTTP Servers...")
    print()

    servers = [
        ("Markdown Processor", "src/mcp/servers/markdown_processor.py", 8000),
        ("DateTime Processor", "src/mcp/servers/datetime_processor.py", 8001),
    ]

    processes = []

    try:
        for name, script, port in servers:
            process = await run_server(name, script, port)
            if process:
                processes.append((name, process, port))

        if not processes:
            print("❌ No servers started successfully")
            return

        print()
        print("🎉 All servers are running!")
        print()
        print("📋 Server URLs:")
        for name, _, port in processes:
            print(f"  • {name}: http://127.0.0.1:{port}/mcp")

        print()
        print("💡 Now you can:")
        print("  1. Update your servers_config.json to use HTTP URLs")
        print("  2. Run your chatbot: streamlit run chatbot.py")
        print("  3. Test the HTTP transport functionality")
        print()
        print("Press Ctrl+C to stop all servers...")

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)

            # Check if any process died
            for name, process, port in processes:
                if process.poll() is not None:
                    print(f"⚠️  {name} stopped unexpectedly")

    except KeyboardInterrupt:
        print("\n🛑 Stopping all servers...")

        for name, process, _ in processes:
            if process.poll() is None:
                print(f"  Stopping {name}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

        print("✅ All servers stopped")


if __name__ == "__main__":
    asyncio.run(main())
