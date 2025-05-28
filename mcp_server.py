from typing import Any, Dict, List
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
# Replace "my_mcp_server" with your desired server name
mcp = FastMCP("my_mcp_server", "My MCP Server", "0.1.0")

# --- Tools ---
# Tools are functions that the LLM can call.
# The @mcp.tool() decorator automatically generates the necessary schema
# from type hints and docstrings.

@mcp.tool()
async def get_server_status() -> Dict[str, Any]:
    """
    Gets the current status of the server.
    Returns a dictionary with server status information.
    """
    return {"status": "running", "version": "0.1.0", "active_users": 5}

@mcp.tool()
async def reverse_string(text: str) -> str:
    """
    Reverses the provided string.

    Args:
        text: The string to reverse.
    
    Returns:
        The reversed string.
    """
    return text[::-1]

# --- Prompts ---
# Prompts are pre-defined templates that can be presented to the user
# or used by the LLM. The MCP Python SDK might offer a specific
# decorator or registration method for prompts.

# Conceptual: The SDK might provide something like @mcp.prompt()
# Or, you might need to register them manually using a mcp.add_prompt() method.

# Example of how a prompt might be defined:
# @mcp.prompt(name="summarize_text_prompt", description="A prompt to summarize long text.")
async def get_summarize_text_prompt(text_to_summarize: str) -> List[Dict[str, Any]]:
    """
    Generates a prompt to summarize the provided text.
    This function would return a list of messages formatted for the LLM.
    """
    return [
        {
            "role": "user",
            "content": {
                "type": "text",
                "text": f"Please summarize the following text concisely:\n\n{text_to_summarize}"
            }
        }
    ]

# If using manual registration, it might look like this (conceptual):
# mcp.register_prompt(
#     name="summarize_text_prompt",
#     description="A prompt to summarize long text.",
#     arguments=[{"name": "text_to_summarize", "description": "The text to be summarized", "required": True}],
#     handler=get_summarize_text_prompt 
# )


# --- Resources ---
# Resources are data that the server can expose to the client/LLM,
# like files or database entries.
# Similar to prompts, the SDK likely provides a way to register resources.

# Conceptual: The SDK might provide an @mcp.resource() decorator for simple cases,
# or methods to define resources with more complex behaviors (e.g., read, subscribe).

# Example of how a static resource might be defined:
# @mcp.resource(uri="resource://server/welcome_message", mime_type="text/plain")
async def get_welcome_message_resource() -> str:
    """
    Provides a static welcome message resource.
    """
    return "Welcome to My MCP Server! We offer various tools and information."

# For dynamic resources or those requiring specific read handlers:
# This is highly conceptual and depends on the SDK's design.
# It might involve creating a class that implements a Resource interface
# or registering handler functions for specific URIs.

# Conceptual registration for a resource:
# async def read_dynamic_data(uri: str, params: Dict[str, Any]) -> str:
#     # Logic to fetch and return data based on URI and params
#     return f"Dynamic data for {uri} with params {params}"

# mcp.register_resource(
#     uri_template="resource://server/dynamic_data/{item_id}",
#     name="Dynamic Data Resource",
#     description="Accesses dynamic data items by ID.",
#     mime_type="application/json", # Or appropriate type
#     read_handler=read_dynamic_data 
#     # Potentially other handlers for list, subscribe etc.
# )


# --- Main execution ---
if __name__ == "__main__":
    # You would add your prompt and resource registrations here
    # if they are not handled by decorators or need specific setup.
    
    # Example (conceptual) for registering the prompt and resource handlers if not using decorators:
    # This assumes methods like `add_prompt_handler` and `add_resource_handler` exist.
    # The actual methods would depend on the mcp SDK.
    
    # mcp.add_prompt_handler("summarize_text_prompt", get_summarize_text_prompt, 
    #                        description="A prompt to summarize long text.", 
    #                        arguments=[{"name": "text_to_summarize", "description": "The text to summarize", "required": True}])
    
    # mcp.add_resource_handler("resource://server/welcome_message", get_welcome_message_resource, mime_type="text/plain")

    print("Starting MCP server...")
    # Run the server (e.g., using stdio transport as in the tutorial)
    # The transport method and other run configurations might vary.
    # Refer to the MCP Python SDK documentation for details.
    mcp.run(transport='stdio')
    print("MCP server stopped.") 