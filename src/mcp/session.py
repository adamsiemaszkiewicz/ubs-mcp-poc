import json
import logging
import re
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

import colorama

from src.mcp.client import MCPClient
from src.model.oai import OpenAIClient
from src.workflow import WorkflowEventType, WorkflowTracer

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# System message template for LLM with tool integration
# Following MCP best practices for tool invocation
SYSTEM_MESSAGE = (
    "You are a helpful assistant with access to these tools:\n\n"
    "{tools_description}\n\n"
    "IMPORTANT INSTRUCTIONS FOR TOOL USAGE:\n\n"
    "1. When you need to use a tool to answer a user's question, respond with ONLY the JSON object below (no additional text):\n"
    "{{\n"
    '    "tool": "tool-name",\n'
    '    "arguments": {{\n'
    '        "argument-name": "value"\n'
    "    }}\n"
    "}}\n\n"
    "2. DO NOT include any conversational text when making a tool call.\n"
    "3. DO NOT include multiple tool calls in a single response.\n"
    "4. After the tool executes, you will receive the results and can then provide a natural, conversational response.\n\n"
    "5. If no tool is needed, reply directly with conversational text.\n\n"
    "When responding after receiving tool results:\n"
    "- Transform the raw data into a natural, conversational response\n"
    "- Keep responses concise but informative\n"
    "- Focus on the most relevant information\n"
    "- Use appropriate context from the user's question\n"
    "- Do not repeat the raw technical data\n\n"
    "Remember: Either respond with ONLY a JSON tool call, OR respond with ONLY conversational text. Never mix both in the same response."
)


@dataclass
class ToolCall:
    """Represents a single tool call execution result.

    Following MCP tools specification, this tracks the complete lifecycle
    of a tool execution including the request, result, and any errors that occurred.

    Attributes:
        tool: Name of the tool that was called
        arguments: Arguments passed to the tool
        result: Result returned by the tool (if successful)
        error: Error message if tool execution failed

    """

    tool: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None

    def is_successful(self) -> bool:
        """Check if the tool call completed successfully.

        Returns:
            True if tool executed without errors and returned a result

        """
        return self.error is None and self.result is not None

    def to_description(self, for_display: bool = False, max_length: int = 200) -> str:
        """Format the tool call execution for display or logging.

        Args:
            for_display: Whether to format for UI display (truncates long content)
            max_length: Maximum length of result/error content when for_display=True

        Returns:
            A formatted string describing the tool call and its outcome

        """
        base_description = f"Tool Name: {self.tool}\n- Arguments: {json.dumps(self.arguments, indent=2)}\n"
        final_description = base_description
        if self.is_successful():
            result_str = str(self.result)[:max_length] if for_display else str(self.result)
            final_description += f"- Tool call result: {result_str}\n"
        else:
            error_str = str(self.error)[:max_length] if for_display else str(self.error)
            final_description += f"- Tool call error: {error_str}\n"
        return final_description


class ChatSession:
    """Orchestrates multi-turn conversations with dynamic MCP tool integration.

    This class implements the MCP Host pattern as defined in the architecture specification
    (mcp-docs/sections/architecture.md). Key responsibilities:

    **MCP Host Responsibilities:**
    - Creates and manages multiple MCP client instances (one per server)
    - Controls client connection permissions and lifecycle
    - Coordinates AI/LLM integration and tool execution
    - Manages context aggregation across multiple servers
    - Enforces security boundaries between servers

    **Multi-Client Architecture:**
    Per MCP specification, each client maintains a 1:1 relationship with a server.
    This session manages multiple clients simultaneously, enabling:
    - Tool discovery across all connected servers
    - Intelligent routing of tool calls to appropriate servers
    - Consolidated tool namespace for the LLM
    - Isolation between different server capabilities

    **Tool Execution Flow:**
    1. LLM generates tool calls in JSON format
    2. Session extracts and validates tool calls
    3. Routes each tool call to the appropriate MCP client/server
    4. Aggregates results and formats them for LLM consumption
    5. LLM generates natural language response based on tool results
    """

    def __init__(self, clients: List[MCPClient], llm_client: OpenAIClient) -> None:
        """Initialize ChatSession with MCP clients and LLM client.

        Args:
            clients: List of initialized MCP clients, each connected to a different server
            llm_client: LLM client for generating responses and tool calls

        Note:
            Following MCP architecture best practices:
            - One client per server (1:1 relationship)
            - Host (this session) manages multiple clients
            - Each client maintains isolated server connection

        """
        self.clients: List[MCPClient] = clients
        self.llm_client: OpenAIClient = llm_client
        self.messages: List[Dict[str, str]] = []
        self._is_initialized: bool = False

        # Tool routing map: tool_name -> client
        # This enables efficient routing of tool calls to the correct server
        self.tool_client_map: Dict[str, MCPClient] = {}

    async def cleanup_clients(self) -> None:
        """Clean up all MCP client resources.

        Properly terminates all MCP client connections following
        the lifecycle management best practices.
        """
        for client in self.clients:
            try:
                await client.cleanup()
            except Exception as e:
                logging.warning(f"Warning during cleanup of client {client.name}: {e}")

    async def initialize(self) -> bool:
        """Initialize all MCP clients and prepare the session.

        Following MCP initialization sequence:
        1. Initialize each MCP client (connects to server, negotiates capabilities)
        2. Discover tools from all servers
        3. Build tool routing map for efficient execution
        4. Create system message with consolidated tool descriptions

        Returns:
            True if initialization successful, False otherwise

        Note:
            This implements the MCP Host pattern where the host manages
            multiple client connections and aggregates their capabilities.

        """
        try:
            if self._is_initialized:
                logging.info("ChatSession already initialized")
                return True

            logging.info(f"Initializing ChatSession with {len(self.clients)} MCP clients")

            # Phase 1: Initialize all MCP clients
            # Each client establishes connection to its server and negotiates capabilities
            for client in self.clients:
                try:
                    await client.initialize()
                    logging.info(f"Successfully initialized client: {client.name}")
                except Exception as e:
                    logging.error(f"Failed to initialize client {client.name}: {e}")
                    await self.cleanup_clients()
                    return False

            # Phase 2: Build tool routing map
            # Discover tools from all servers and create unified tool namespace
            await self._build_tool_routing_map()

            # Phase 3: Create system message with all available tools
            # This provides the LLM with complete visibility of available capabilities
            await self._create_system_message()

            self._is_initialized = True
            logging.info(f"ChatSession initialized successfully with {len(self.tool_client_map)} total tools")
            return True

        except Exception as e:
            logging.error(f"ChatSession initialization error: {e}")
            await self.cleanup_clients()
            return False

    async def _build_tool_routing_map(self) -> None:
        """Build the tool-to-client routing map.

        Creates a unified tool namespace by collecting tools from all connected
        MCP servers. Handles tool name conflicts by logging warnings.

        Following MCP design principle: "Servers should be highly composable"
        - Each server provides focused functionality in isolation
        - Multiple servers can be combined seamlessly
        - Host manages tool namespace conflicts
        """
        self.tool_client_map = {}

        for client in self.clients:
            try:
                tools = await client.list_tools()
                for tool in tools:
                    if tool.name in self.tool_client_map:
                        # Handle tool name conflicts
                        existing_client = self.tool_client_map[tool.name]
                        logging.warning(
                            f"Tool name conflict: '{tool.name}' exists in both "
                            f"'{existing_client.name}' and '{client.name}'. "
                            f"Using tool from '{existing_client.name}'"
                        )
                        continue

                    self.tool_client_map[tool.name] = client
                    logging.debug(f"Mapped tool '{tool.name}' to client '{client.name}'")

            except Exception as e:
                logging.error(f"Error listing tools from client {client.name}: {e}")

    async def _create_system_message(self) -> None:
        """Create system message with consolidated tool descriptions.

        Builds a comprehensive system prompt that includes:
        - Descriptions of all available tools from all servers
        - Instructions for tool invocation format
        - Guidelines for natural language response generation
        """
        # Collect all tools from all clients for LLM visibility
        all_tools = []
        for client in self.clients:
            try:
                tools = await client.list_tools()
                all_tools.extend(tools)
            except Exception as e:
                logging.error(f"Error collecting tools from {client.name}: {e}")

        # Format tool descriptions for LLM understanding
        tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])
        system_message = SYSTEM_MESSAGE.format(tools_description=tools_description)

        # Initialize conversation with system message
        self.messages = [{"role": "system", "content": system_message}]
        logging.info(f"System message created with {len(all_tools)} tools from {len(self.clients)} servers")

    def _extract_tool_calls(self, llm_response: str) -> List[Dict[str, Any]]:
        """Extract tool call JSON objects from LLM response.

        Handles multiple response formats following MCP tool invocation patterns:
        1. Single JSON object tool call
        2. Multiple JSON objects in response
        3. Mixed text and JSON content

        Args:
            llm_response: Raw response text from the LLM

        Returns:
            List of extracted tool call dictionaries with 'tool' and 'arguments' keys

        Note:
            This implements robust parsing to handle various LLM response formats
            while maintaining compatibility with MCP tool call specification.

        """
        # Attempt 1: Parse entire response as single JSON tool call
        try:
            tool_call = json.loads(llm_response.strip())
            if isinstance(tool_call, dict) and "tool" in tool_call and "arguments" in tool_call:
                logging.debug("Extracted single tool call from response")
                return [tool_call]
        except json.JSONDecodeError:
            pass

        # Attempt 2: Extract multiple JSON objects using regex
        tool_calls = []
        # Improved regex pattern to match nested JSON objects
        json_pattern = r"({[^{}]*(?:{[^{}]*}[^{}]*)*})"
        json_matches = re.finditer(json_pattern, llm_response)

        for match in json_matches:
            try:
                json_obj = json.loads(match.group(0))
                if isinstance(json_obj, dict) and "tool" in json_obj and "arguments" in json_obj:
                    tool_calls.append(json_obj)
                    logging.debug(f"Extracted tool call: {json_obj['tool']}")
            except json.JSONDecodeError:
                continue

        if tool_calls:
            logging.info(f"Extracted {len(tool_calls)} tool calls from LLM response")

        return tool_calls

    async def _execute_tool_call(self, tool_call_data: Dict[str, Any]) -> ToolCall:
        """Execute a single tool call by routing to the appropriate MCP client.

        Args:
            tool_call_data: Dictionary containing 'tool' name and 'arguments'

        Returns:
            ToolCall object with execution result or error

        Following MCP execution pattern:
        1. Validate tool exists in routing map
        2. Route to appropriate MCP client based on tool ownership
        3. Execute via MCP tools/call protocol
        4. Return structured result for further processing

        """
        tool_name = tool_call_data["tool"]
        arguments = tool_call_data["arguments"]

        tool_call = ToolCall(tool=tool_name, arguments=arguments)

        # Route tool call to appropriate client using our routing map
        if tool_name in self.tool_client_map:
            client = self.tool_client_map[tool_name]
            try:
                logging.info(f"Routing tool '{tool_name}' to server '{client.name}'")
                result = await client.execute_tool(tool_name, arguments)
                tool_call.result = result
                logging.info(f"Tool '{tool_name}' executed successfully")
                return tool_call

            except Exception as e:
                error_msg = f"Error executing tool '{tool_name}' on server '{client.name}': {str(e)}"
                logging.error(error_msg)
                tool_call.error = error_msg
                return tool_call

        # Tool not found in any connected server
        error_msg = f"Tool '{tool_name}' not found in any connected MCP server. Available tools: {list(self.tool_client_map.keys())}"
        logging.error(error_msg)
        tool_call.error = error_msg
        return tool_call

    async def process_tool_calls(
        self,
        llm_response: str,
        tool_call_data_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[ToolCall], bool]:
        """Process all tool calls in an LLM response.

        Orchestrates the complete tool execution pipeline:
        1. Extract tool calls from response (if not provided)
        2. Execute each tool call via appropriate MCP client
        3. Collect and return all results

        Args:
            llm_response: Raw LLM response text
            tool_call_data_list: Optional pre-extracted tool calls

        Returns:
            Tuple of (list of ToolCall results, bool indicating if any tools were executed)

        Note:
            This implements parallel tool execution where possible while maintaining
            proper error isolation between different server connections.

        """
        # Extract tool calls from response if not provided
        if tool_call_data_list is None:
            tool_call_data_list = self._extract_tool_calls(llm_response)

        if not tool_call_data_list:
            return [], False

        logging.info(f"Processing {len(tool_call_data_list)} tool calls")

        # Execute each tool call
        # Note: Currently sequential execution, could be made parallel for independent tools
        tool_calls = []
        for i, tool_call_data in enumerate(tool_call_data_list, 1):
            logging.debug(f"Executing tool call {i}/{len(tool_call_data_list)}: {tool_call_data['tool']}")
            tool_call = await self._execute_tool_call(tool_call_data)
            tool_calls.append(tool_call)

        # Log execution summary
        successful_calls = [tc for tc in tool_calls if tc.is_successful()]
        failed_calls = [tc for tc in tool_calls if not tc.is_successful()]

        logging.info(f"Tool execution complete: {len(successful_calls)} successful, {len(failed_calls)} failed")

        return tool_calls, True

    def _format_tool_results(
        self,
        tool_calls: List[ToolCall],
        for_display: bool = False,
        max_length: int = 200,
    ) -> str:
        """Format tool execution results for LLM consumption.

        Creates a structured text representation of tool results that helps
        the LLM understand what was executed and what the outcomes were.

        Args:
            tool_calls: List of ToolCall objects with execution results
            for_display: Whether to format for UI display (affects truncation)
            max_length: Maximum length of individual result content

        Returns:
            Formatted string describing all tool executions and their results

        """
        if not tool_calls:
            return "No tools were executed."

        results = []
        for i, call in enumerate(tool_calls, 1):
            result_text = f"Tool Call {i}:\n"
            result_text += call.to_description(for_display, max_length)
            results.append(result_text)

        formatted_results = "\n".join(results)
        logging.debug(f"Formatted {len(tool_calls)} tool results for LLM")

        return formatted_results

    async def send_message(
        self,
        user_message: str,
        auto_process_tools: bool = True,
        show_workflow: bool = False,
        max_iterations: int = 10,
    ) -> str:
        """Send message and get response, optionally auto-process tool calls.

        Args:
            user_message: The user's message
            auto_process_tools: Whether to auto-process tool calls
            show_workflow: Whether to show the workflow
            max_iterations: Maximum number of tool iterations (default: 10)

        Returns:
            The final response text

        """
        if not self._is_initialized:
            success = await self.initialize()
            if not success:
                return "Failed to initialize chat session"

        # Initialize the workflow tracer
        self.workflow_tracer = WorkflowTracer()

        # Record user query
        self.workflow_tracer.add_event(
            WorkflowEventType.USER_QUERY,
            user_message[:50] if len(user_message) > 50 else user_message,
        )

        self.messages.append({"role": "user", "content": user_message})

        # Record LLM thinking
        self.workflow_tracer.add_event(WorkflowEventType.LLM_THINKING, "LLM is processing your query...")

        # Get LLM response
        llm_response = self.llm_client.get_response(self.messages)

        # Record LLM response
        self.workflow_tracer.add_event(
            WorkflowEventType.LLM_RESPONSE,
            llm_response[:50] if len(llm_response) > 50 else llm_response,
        )

        self.messages.append({"role": "assistant", "content": llm_response})
        logging.info(f"\n{colorama.Fore.YELLOW}[Debug] LLM Response: {llm_response}{colorama.Style.RESET_ALL}")

        if not auto_process_tools:
            # Record final response
            self.workflow_tracer.add_event(
                WorkflowEventType.FINAL_RESPONSE,
                "Direct response without tool processing",
            )
            # Output formatted workflow
            if show_workflow:
                print(self.workflow_tracer.render_tree_workflow())
            return llm_response

        # Automatically process tool calls
        tool_iteration = 0
        while tool_iteration < max_iterations:
            tool_iteration += 1
            tool_calls, has_tools = await self.process_tool_calls(llm_response)

            if not has_tools:
                # Record final response
                self.workflow_tracer.add_event(
                    WorkflowEventType.FINAL_RESPONSE,
                    f"Final response after {tool_iteration - 1} tool iterations",
                )
                # Output formatted workflow
                if show_workflow:
                    print(self.workflow_tracer.render_tree_workflow())
                return llm_response

            # Record tool calls
            for i, tool_call in enumerate(tool_calls):
                # Record tool call request
                self.workflow_tracer.add_event(
                    WorkflowEventType.TOOL_CALL,
                    f"Call {i + 1}: {tool_call.tool}",
                    {"tool_name": tool_call.tool, "arguments": tool_call.arguments},
                )

                # Record tool execution
                self.workflow_tracer.add_event(WorkflowEventType.TOOL_EXECUTION, f"Executing {tool_call.tool}...")

                # Record tool result
                success = tool_call.is_successful()
                self.workflow_tracer.add_event(
                    WorkflowEventType.TOOL_RESULT,
                    "Success" if success else f"Error: {tool_call.error}",
                    {
                        "success": success,
                        "result": str(tool_call.result)[:100] if success else None,
                    },
                )

            # Format tool results and add to message history
            tool_results = self._format_tool_results(tool_calls)
            self.messages.append({"role": "system", "content": tool_results})
            tool_result_formatted = self._format_tool_results(tool_calls, for_display=True)
            logging.info(
                f"\n{colorama.Fore.MAGENTA}[Debug] Tool Results: {tool_result_formatted}{colorama.Style.RESET_ALL}"
            )

            # Record LLM thinking again
            self.workflow_tracer.add_event(
                WorkflowEventType.LLM_THINKING,
                f"LLM processing tool results (iteration {tool_iteration})...",
            )

            # Get next response
            llm_response = self.llm_client.get_response(self.messages)

            # Record LLM response
            self.workflow_tracer.add_event(
                WorkflowEventType.LLM_RESPONSE,
                llm_response[:50] if len(llm_response) > 50 else llm_response,
            )

            self.messages.append({"role": "assistant", "content": llm_response})
            logging.info(f"\n{colorama.Fore.YELLOW}[Debug] LLM Response: {llm_response}{colorama.Style.RESET_ALL}")

            # Check if next response still contains tool calls
            next_tool_calls = self._extract_tool_calls(llm_response)
            if not next_tool_calls:
                # Record final response
                self.workflow_tracer.add_event(
                    WorkflowEventType.FINAL_RESPONSE,
                    f"Final response after {tool_iteration} tool iterations",
                )
                # Output formatted workflow
                if show_workflow:
                    print(self.workflow_tracer.render_tree_workflow())
                return llm_response

        # If we reach max_iterations, return the last response
        self.workflow_tracer.add_event(
            WorkflowEventType.FINAL_RESPONSE,
            f"Max iterations ({max_iterations}) reached",
        )
        if show_workflow:
            print(self.workflow_tracer.render_tree_workflow())
        return llm_response

    async def send_message_stream(
        self,
        user_message: str,
        auto_process_tools: bool = True,
        show_workflow: bool = False,
        max_iterations: int = 10,
    ) -> AsyncGenerator[Union[str, Tuple[str, str]], None]:
        """Send message and get streaming response, with optional tool processing.

        Args:
            user_message: The user's message
            auto_process_tools: Whether to auto-process tool calls
            show_workflow: Whether to show the workflow
            max_iterations: Maximum number of tool iterations (default: 10)

        Yields:
            Response text chunks or tuples of (status, text_chunk)

        """
        if not self._is_initialized:
            success = await self.initialize()
            if not success:
                yield ("error", "Failed to initialize chat session")
                return

        # Initialize the workflow tracer
        self.workflow_tracer = WorkflowTracer()

        # Record user query
        self.workflow_tracer.add_event(
            WorkflowEventType.USER_QUERY,
            user_message[:50] if len(user_message) > 50 else user_message,
        )

        self.messages.append({"role": "user", "content": user_message})

        # Record LLM thinking
        self.workflow_tracer.add_event(WorkflowEventType.LLM_THINKING, "LLM is processing your query...")

        #### Get initial response stream ####
        yield ("status", "Thinking...")
        response_chunks = []
        for chunk in self.llm_client.get_stream_response(self.messages):
            response_chunks.append(chunk)
            yield ("response", chunk)
        #####################################

        llm_response = "".join(response_chunks)

        # Record LLM response
        self.workflow_tracer.add_event(
            WorkflowEventType.LLM_RESPONSE,
            llm_response[:50] if len(llm_response) > 50 else llm_response,
        )

        self.messages.append({"role": "assistant", "content": llm_response})

        if not auto_process_tools:
            # Record final response
            self.workflow_tracer.add_event(
                WorkflowEventType.FINAL_RESPONSE,
                "Direct response without tool processing",
            )
            if show_workflow:
                print(self.workflow_tracer.render_tree_workflow())
            return

        # Process tool calls
        iteration = 0
        while iteration < max_iterations:
            # Extract tool call data
            tool_call_data_list = self._extract_tool_calls(llm_response)

            if not tool_call_data_list:
                # No tool calls, return final result
                self.workflow_tracer.add_event(
                    WorkflowEventType.FINAL_RESPONSE,
                    f"Final response after {iteration} tool iterations",
                )
                if show_workflow:
                    print(self.workflow_tracer.render_tree_workflow())
                return

            # Process each tool call separately, and pass detailed information to the UI
            tool_calls = []
            for idx, tool_call_data in enumerate(tool_call_data_list):
                tool_name = tool_call_data["tool"]
                arguments = tool_call_data["arguments"]

                # Pass tool name and arguments to the UI
                yield ("tool_call", tool_name)
                yield ("tool_arguments", json.dumps(arguments))

                # Record tool call request
                self.workflow_tracer.add_event(
                    WorkflowEventType.TOOL_CALL,
                    f"Call {idx + 1}: {tool_name}",
                    {"tool_name": tool_name, "arguments": arguments},
                )

                # Pass tool execution status to the UI
                yield ("tool_execution", f"Executing {tool_name}...")

                # Record tool execution
                self.workflow_tracer.add_event(WorkflowEventType.TOOL_EXECUTION, f"Executing {tool_name}...")

                # Execute tool call
                tool_call = await self._execute_tool_call(tool_call_data)
                tool_calls.append(tool_call)

                # Record tool result
                success = tool_call.is_successful()
                self.workflow_tracer.add_event(
                    WorkflowEventType.TOOL_RESULT,
                    "Success" if success else f"Error: {tool_call.error}",
                    {
                        "success": success,
                        "result": str(tool_call.result)[:100] if success else None,
                    },
                )

                # Pass tool result status to the UI
                yield (
                    "tool_results",
                    json.dumps(
                        {
                            "success": success,
                            "result": str(tool_call.result) if success else str(tool_call.error),
                        }
                    ),
                )

            # Format tool results and add to message history
            tool_results = self._format_tool_results(tool_calls)
            self.messages.append({"role": "system", "content": tool_results})

            # Record LLM thinking
            self.workflow_tracer.add_event(
                WorkflowEventType.LLM_THINKING,
                f"LLM processing tool results (iteration {iteration + 1})...",
            )

            # Get next response stream
            yield ("status", "Processing results...")
            response_chunks = []
            for chunk in self.llm_client.get_stream_response(self.messages):
                response_chunks.append(chunk)
                yield ("response", chunk)

            llm_response = "".join(response_chunks)

            # Record LLM response
            self.workflow_tracer.add_event(
                WorkflowEventType.LLM_RESPONSE,
                llm_response[:50] if len(llm_response) > 50 else llm_response,
            )

            self.messages.append({"role": "assistant", "content": llm_response})

            # Check if next response still contains tool calls
            next_tool_calls = self._extract_tool_calls(llm_response)
            if not next_tool_calls:
                # Record final response
                self.workflow_tracer.add_event(
                    WorkflowEventType.FINAL_RESPONSE,
                    f"Final response after {iteration + 1} tool iterations",
                )
                if show_workflow:
                    print(self.workflow_tracer.render_tree_workflow())
                return

            iteration += 1
