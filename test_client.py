mport asyncio
import os

from src.mcp.client import MCPClient


async def test_markdown_operations():
    """Test the markdown processor operations."""
    # Initialize client
    client = MCPClient()

    try:
        # Connect to the markdown processor server
        await client.connect_to_server("src/mcp/servers/markdown_processor.py")

        # Test queries
        test_queries = [
            "Create a new markdown file in the ./tmp directory called 'test.md' with content about Python programming"
        ]

        print("Testing MCP Client with Markdown Processor")
        print("=" * 50)

        for i, query in enumerate(test_queries, 1):
            print(f"\nTest {i}: {query}")
            print("-" * 30)

            try:
                response = await client.process_query(query)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error processing query: {e}")

    finally:
        await client.cleanup()


if __name__ == "__main__":
    # Ensure we have the required environment variable
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        exit(1)

    asyncio.run(test_markdown_operations())
