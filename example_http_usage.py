import asyncio

from src.mcp.client import MCPClient


async def test_datetime_server():
    """Test the datetime server via HTTP."""
    print("🕒 Testing DateTime Server via HTTP...")

    datetime_config = {"url": "http://127.0.0.1:8001/mcp", "timeout": 30}

    async with MCPClient("datetime_http", datetime_config) as client:
        print(f"✅ Connected to {client.name}")

        tools = await client.list_tools()
        print(f"📋 Available tools: {[tool.name for tool in tools]}")

        if tools:
            # Test getting local datetime
            if any(tool.name == "get_local_datetime" for tool in tools):
                print("⏰ Getting local datetime...")
                result = await client.execute_tool("get_local_datetime", {})
                print(f"📅 Current time: {result}")

            # Test timezone datetime
            if any(tool.name == "get_datetime_for_timezone" for tool in tools):
                print("🌍 Getting UTC time...")
                result = await client.execute_tool("get_datetime_for_timezone", {"timezone": "UTC"})
                print(f"🌐 UTC time: {result}")


async def test_markdown_server():
    """Test the markdown server via HTTP."""
    print("\n📝 Testing Markdown Server via HTTP...")

    markdown_config = {"url": "http://127.0.0.1:8000/mcp", "timeout": 30}

    async with MCPClient("markdown_http", markdown_config) as client:
        print(f"✅ Connected to {client.name}")

        tools = await client.list_tools()
        print(f"📋 Available tools: {[tool.name for tool in tools]}")

        if tools:
            # Test reading markdown files
            if any(tool.name == "read_markdown_file" for tool in tools):
                print("📖 Reading markdown files from root directory...")
                try:
                    result = await client.execute_tool("read_markdown_file", {"directory_path": "."})
                    print(f"📄 Found markdown content (first 200 chars): {str(result)[:200]}...")
                except Exception as e:
                    print(f"⚠️  Error reading markdown: {e}")





async def main():
    """Run all examples."""
    print("🚀 Testing MCP HTTP Transport vs STDIO Transport")
    print("=" * 60)

    try:
        await test_datetime_server()
        await test_markdown_server()

        print("\n✅ All tests completed successfully!")
        print("\n💡 Key Differences:")
        print("  • HTTP: Servers run as persistent processes")
        print("  • STDIO: Servers start/stop for each connection")
        print("  • HTTP: Better for web deployments")
        print("  • STDIO: Better for local tools")

    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        print("\n💡 Make sure to run the servers first:")
        print("   python run_http_servers.py")


if __name__ == "__main__":
    asyncio.run(main())
