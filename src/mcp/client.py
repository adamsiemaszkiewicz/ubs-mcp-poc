import asyncio
import logging
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPTool:
    """Represents a MCP tool with its properties and formatting.

    Following the MCP tools specification (mcp-docs/sections/tools.md),
    each tool has a unique name, description, and input schema that defines
    expected parameters using JSON Schema.

    Attributes:
        name: Unique identifier for the tool
        description: Human-readable description of functionality
        input_schema: JSON Schema defining expected parameters

    """

    def __init__(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        """Initialize MCPTool with required properties.

        Args:
            name: Unique tool identifier (must be unique across all connected servers)
            description: Human-readable description for the LLM to understand tool purpose
            input_schema: JSON Schema object defining tool parameters and types

        """
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM consumption.

        Creates a structured text representation that helps the LLM understand:
        - What the tool does (description)
        - What parameters it accepts (from schema properties)
        - Which parameters are required vs optional

        Returns:
            A formatted string describing the tool for LLM prompt inclusion.

        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class MCPClient:
    """MCP Client implementation following the official specification.

    Per MCP architecture (mcp-docs/sections/architecture.md):
    - Each client maintains a 1:1 stateful session with ONE server
    - Clients handle protocol negotiation and capability exchange
    - Clients route protocol messages bidirectionally
    - Clients maintain security boundaries between servers

    This implementation uses STDIO transport for local server communication
    and manages the complete client lifecycle including initialization,
    tool discovery, execution, and cleanup.

    Key Design Principles:
    1. One client per server (1:1 relationship as per MCP spec)
    2. Proper resource management using AsyncExitStack
    3. Comprehensive error handling with retries
    4. Clean separation of concerns
    """

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        """Initialize MCP Client for a specific server.

        Args:
            name: Unique name for this client/server connection
            config: Server configuration containing:
                - command: Executable command (e.g., "python", "node", "npx")
                - args: Command line arguments for the server
                - env: Optional environment variables for the server process

        """
        self.name: str = name
        self.config: dict[str, Any] = config

        # Connection state management
        self.stdio_context: Any | None = None
        self.session: Optional[ClientSession] = None
        self._is_initialized: bool = False

        # Resource management
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

        # Capability tracking (per MCP capability negotiation)
        self._server_capabilities: Optional[Dict[str, Any]] = None
        self._available_tools: List[MCPTool] = []

    async def initialize(self) -> None:
        """Initialize the MCP server connection with capability negotiation.

        Following MCP lifecycle specification (mcp-docs/sections/lifecycle.md):
        1. Establish transport connection (STDIO)
        2. Create client session
        3. Perform MCP initialization handshake
        4. Negotiate capabilities between client and server
        5. Discover available tools

        Raises:
            ValueError: If server command is invalid or not found
            RuntimeError: If connection or initialization fails
            Exception: For any other initialization errors

        """
        if self._is_initialized:
            logging.info(f"Client {self.name} already initialized")
            return

        # Resolve server command (handle npx and other commands)
        command = shutil.which("npx") if self.config["command"] == "npx" else self.config["command"]
        if command is None:
            raise ValueError(f"Server command '{self.config['command']}' not found in PATH")

        # Prepare server parameters for STDIO transport
        server_params = StdioServerParameters(
            command=command,
            args=self.config["args"],
            env={**os.environ, **self.config["env"]} if self.config.get("env") else None,
        )

        try:
            # Establish STDIO transport connection
            logging.info(f"Connecting to MCP server {self.name} via STDIO transport")
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport

            # Create and initialize MCP client session
            session = await self.exit_stack.enter_async_context(ClientSession(read, write))

            # Perform MCP initialization handshake
            # This exchanges protocol versions and capabilities
            init_result = await session.initialize()
            logging.info(f"MCP session initialized for {self.name}")

            # Store session and capabilities
            self.session = session
            self._server_capabilities = getattr(init_result, "capabilities", {})

            # Discovery phase: Get available tools from server
            await self._discover_tools()

            self._is_initialized = True
            logging.info(f"Client {self.name} successfully initialized with {len(self._available_tools)} tools")

        except Exception as e:
            logging.error(f"Error initializing MCP server {self.name}: {e}")
            await self.cleanup()
            raise RuntimeError(f"Failed to initialize MCP client {self.name}: {e}") from e

    async def _discover_tools(self) -> None:
        """Discover available tools from the connected server.

        Following MCP tools specification (mcp-docs/sections/tools.md):
        - Sends tools/list request to server
        - Processes tool definitions with names, descriptions, and schemas
        - Caches tools for efficient access

        Raises:
            RuntimeError: If session is not initialized

        """
        if not self.session:
            raise RuntimeError(f"Cannot discover tools: server {self.name} not initialized")

        try:
            logging.debug(f"Discovering tools for server {self.name}")
            tools_response = await self.session.list_tools()

            # Process tools response - handle different response formats
            tools = []
            for item in tools_response:
                if isinstance(item, tuple) and item[0] == "tools":
                    # Standard tools response format
                    for tool in item[1]:
                        mcp_tool = MCPTool(tool.name, tool.description, tool.inputSchema)
                        tools.append(mcp_tool)
                        logging.debug(f"Discovered tool: {tool.name}")

            self._available_tools = tools
            logging.info(f"Server {self.name} provides {len(tools)} tools")

        except Exception as e:
            logging.error(f"Error discovering tools from {self.name}: {e}")
            raise

    async def list_tools(self) -> List[MCPTool]:
        """Get list of available tools from the server.

        Returns cached tool list from initialization. Tools are discovered
        during the initialization phase to avoid repeated server calls.

        Returns:
            List of MCPTool objects with tool metadata

        Raises:
            RuntimeError: If the server is not initialized

        """
        if not self._is_initialized or not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        return self._available_tools.copy()  # Return copy to prevent external modification

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism and comprehensive error handling.

        Following MCP tools specification (mcp-docs/sections/tools.md):
        - Sends tools/call request with tool name and arguments
        - Handles both successful results and error responses
        - Implements exponential backoff for retry attempts
        - Validates tool existence before execution

        Args:
            tool_name: Name of the tool to execute (must exist on this server)
            arguments: Tool arguments matching the tool's input schema
            retries: Number of retry attempts for failed executions
            delay: Initial delay between retries (exponential backoff)

        Returns:
            Tool execution result from the server

        Raises:
            RuntimeError: If server is not initialized
            ValueError: If tool doesn't exist on this server
            Exception: If tool execution fails after all retries

        """
        if not self._is_initialized or not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        # Validate tool exists on this server
        available_tool_names = [tool.name for tool in self._available_tools]
        if tool_name not in available_tool_names:
            raise ValueError(
                f"Tool '{tool_name}' not available on server {self.name}. Available tools: {available_tool_names}"
            )

        attempt = 0
        last_exception = None

        while attempt < retries:
            try:
                logging.info(f"Executing tool '{tool_name}' on server {self.name} (attempt {attempt + 1})")
                logging.debug(f"Tool arguments: {arguments}")

                # Execute tool via MCP tools/call request
                result = await self.session.call_tool(tool_name, arguments)

                logging.info(f"Tool '{tool_name}' executed successfully")
                return result

            except Exception as e:
                attempt += 1
                last_exception = e

                if attempt < retries:
                    # Exponential backoff for retries
                    current_delay = delay * (2 ** (attempt - 1))
                    logging.warning(
                        f"Tool execution failed: {e}. Retrying in {current_delay}s (attempt {attempt} of {retries})"
                    )
                    await asyncio.sleep(current_delay)
                else:
                    logging.error(f"Tool '{tool_name}' execution failed after {retries} attempts")

        # All retries exhausted
        raise Exception(f"Tool execution failed after {retries} attempts. Last error: {last_exception}")

    async def get_server_capabilities(self) -> Optional[Dict[str, Any]]:
        """Get the server's declared capabilities.

        Returns the capabilities negotiated during initialization.
        Useful for checking what features the server supports.

        Returns:
            Dictionary of server capabilities or None if not initialized

        """
        return self._server_capabilities.copy() if self._server_capabilities else None

    async def cleanup(self) -> None:
        """Clean up server resources and connections.

        Properly closes the MCP session and cleans up transport resources.
        Uses lock to prevent concurrent cleanup calls.

        Following MCP lifecycle best practices:
        - Graceful session termination
        - Resource cleanup via AsyncExitStack
        - Error logging without raising (cleanup should not fail)
        """
        async with self._cleanup_lock:
            try:
                if self._is_initialized:
                    logging.info(f"Cleaning up MCP client {self.name}")

                # Close AsyncExitStack which handles all managed resources
                await self.exit_stack.aclose()

                # Reset state
                self.session = None
                self.stdio_context = None
                self._is_initialized = False
                self._server_capabilities = None
                self._available_tools = []

                logging.debug(f"Cleanup completed for client {self.name}")

            except Exception as e:
                # Log but don't raise - cleanup should be best effort
                logging.error(f"Error during cleanup of server {self.name}: {e}")

    async def __aenter__(self):
        """Async context manager entry.

        Automatically initializes the client when entering the context.
        """
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit.

        Automatically cleans up resources when exiting the context.
        """
        await self.cleanup()

    def __repr__(self) -> str:
        """String representation for debugging."""
        status = "initialized" if self._is_initialized else "not initialized"
        return f"MCPClient(name='{self.name}', status='{status}', tools={len(self._available_tools)})"
