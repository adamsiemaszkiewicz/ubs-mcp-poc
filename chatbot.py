import asyncio
import json
import os
import re
import sys
from contextlib import AsyncExitStack
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, get_args

import streamlit as st

from src.config import Configuration
from src.consts import ROOT_DIR
from src.mcp.client import MCPClient, MCPTool
from src.mcp.session import ChatSession
from src.model.factory import LLMProvider, create_llm_client

if sys.platform == "win32":
    # The default event loop policy for Windows is SelectorEventLoop
    # which may cause the mcp loading error.
    # Change to WindowsSelectorEventLoopPolicy to fix.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="MCP Declarative Agent", layout="wide")
st.title("🤖 MCP Declarative Agent")
st.caption("Configure and chat with a Declarative Agent capable of using MCP tools.")

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "llm_provider" not in st.session_state:
    st.session_state.llm_provider = "openai"

if "chatbot_config" not in st.session_state:
    st.session_state.chatbot_config = Configuration()

if "mcp_tools_cache" not in st.session_state:
    st.session_state.mcp_tools_cache = {}

# Add state for declarative agent configuration
if "agent_configured" not in st.session_state:
    st.session_state.agent_configured = False
if "agent_name" not in st.session_state:
    st.session_state.agent_name = ""
if "agent_model" not in st.session_state:
    st.session_state.agent_model = ""
if "agent_provider" not in st.session_state:
    st.session_state.agent_provider = "openai"
if "selected_tools" not in st.session_state:
    st.session_state.selected_tools = set()  # Set of tool names
if "agent_description" not in st.session_state:
    st.session_state.agent_description = ""
if "agent_system_prompt" not in st.session_state:
    st.session_state.agent_system_prompt = ""

# Add state for chat session and config tracking
if "chat_session" not in st.session_state:
    st.session_state.chat_session = None
if "session_config_hash" not in st.session_state:
    st.session_state.session_config_hash = None
if "active_mcp_clients" not in st.session_state:
    st.session_state.active_mcp_clients = []  # Track active clients outside stack
if "mcp_client_stack" not in st.session_state:
    st.session_state.mcp_client_stack = None  # Store the stack itself
if "history_messages" not in st.session_state:
    st.session_state.history_messages = []

# --- Constants ---
WORKFLOW_ICONS = {
    "USER_QUERY": "👤",
    "LLM_THINKING": "☁️",
    "LLM_RESPONSE": "💬",
    "TOOL_CALL": "🔧",
    "TOOL_EXECUTION": "⚡️",
    "TOOL_RESULT": "📊",
    "FINAL_STATUS": "✅",
    "ERROR": "❌",
}


# --- DataClass for Workflow Step ---
@dataclass
class WorkflowStep:
    """Workflow step class for tracking chatbot interactions."""

    type: str
    content: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


async def get_mcp_tools(force_refresh: bool = False) -> dict[str, list[MCPTool]]:
    """Get MCP tools from cache or by initializing clients."""
    if not force_refresh and st.session_state.mcp_tools_cache:
        return st.session_state.mcp_tools_cache

    tools_dict = {}
    config = st.session_state.chatbot_config
    server_config_path = ROOT_DIR / "servers_config.json"
    print(server_config_path)
    if not os.path.exists(server_config_path):
        st.sidebar.warning("MCP Server config file not found. No tools loaded.")
        st.session_state.mcp_tools_cache = {}
        return {}

    try:
        server_config = config.load_config(str(server_config_path))
    except Exception as e:
        st.sidebar.error(f"Error loading MCP server config: {e}")
        st.session_state.mcp_tools_cache = {}
        return {}

    async with AsyncExitStack() as stack:
        if "mcpServers" not in server_config:
            st.sidebar.error("Invalid MCP server config format: 'mcpServers' key missing.")
            st.session_state.mcp_tools_cache = {}
            return {}

        for name, srv_config in server_config["mcpServers"].items():
            try:
                client = MCPClient(name, srv_config)
                await stack.enter_async_context(client)
                tools = await client.list_tools()
                tools_dict[name] = tools
            except Exception as e:
                st.sidebar.error(f"Error fetching tools from {name}: {e}")

    st.session_state.mcp_tools_cache = tools_dict
    return tools_dict


def extract_json_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Extract tool call JSON objects from text using robust pattern matching.

    Uses similar logic to ChatSession._extract_tool_calls but adapted for our needs.

    Args:
        text: Text possibly containing JSON tool calls

    Returns:
        Tuple of (list of extracted tool calls, cleaned text without JSON)

    """
    tool_calls = []
    cleaned_text = text
    json_parsed = False

    # Try to parse the entire text as a single JSON array of
    # tool calls or a single tool call object
    try:
        data = json.loads(text.strip())
        if isinstance(data, list):  # Check if it's a list of tool calls
            valid_tools = True
            for item in data:
                if not (isinstance(item, dict) and "tool" in item and "arguments" in item):
                    valid_tools = False
                    break
            if valid_tools:
                tool_calls.extend(data)
                json_parsed = True
        elif isinstance(data, dict) and "tool" in data and "arguments" in data:  # Check if it's a single tool call
            tool_calls.append(data)
            json_parsed = True

        if json_parsed:
            return (
                tool_calls,
                "",
            )  # Return empty string as cleaned text if parsing was successful

    except json.JSONDecodeError:
        pass  # Proceed to regex matching if direct parsing fails

    # Regex pattern to find potential JSON objects (might include tool calls)
    # This pattern tries to find JSON objects starting with '{' and ending with '}'
    json_pattern = r"\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}"
    matches = list(re.finditer(json_pattern, text))
    extracted_indices: set[tuple[int, int]] = set()

    for match in matches:
        start, end = match.span()
        # Avoid processing overlapping matches
        if any(start < prev_end and end > prev_start for prev_start, prev_end in extracted_indices):
            continue

        json_str = match.group(0)
        try:
            obj = json.loads(json_str)
            # Check if the parsed object looks like a tool call
            if isinstance(obj, dict) and "tool" in obj and "arguments" in obj:
                tool_calls.append(obj)
                # Mark this region as extracted
                extracted_indices.add((start, end))
        except json.JSONDecodeError:
            # Ignore parts that are not valid JSON or not tool calls
            pass

    # Build the cleaned text by removing the extracted JSON parts
    if extracted_indices:
        cleaned_parts = []
        last_end = 0
        for start, end in sorted(list(extracted_indices)):
            cleaned_parts.append(text[last_end:start])
            last_end = end
        cleaned_parts.append(text[last_end:])
        cleaned_text = "".join(cleaned_parts).strip()
    else:
        # If no JSON tool calls were extracted via regex,
        # the original text is the cleaned text
        cleaned_text = text

    return tool_calls, cleaned_text


def render_workflow(steps: list[WorkflowStep], container: Optional[Any] = None) -> None:
    """Render workflow steps, placing each tool call sequence in its own expander."""
    if not steps:
        return

    target = container if container else st

    rendered_indices = set()

    # Iterate through steps to render them sequentially
    for i, step in enumerate(steps):
        if i in rendered_indices:
            continue

        step_type = step.type

        if step_type == "TOOL_CALL":
            # Start of a new tool call sequence
            tool_name = step.details.get("tool_name", "Unknown Tool")
            expander_title = f"{WORKFLOW_ICONS['TOOL_CALL']} Tool Call: {tool_name}"
            with target.expander(expander_title, expanded=False):
                # Display arguments
                arguments = step.details.get("arguments", {})
                st.write("**Arguments:**")
                if isinstance(arguments, str) and arguments == "Pending...":
                    st.write("Preparing arguments...")
                elif isinstance(arguments, dict):
                    if arguments:
                        for key, value in arguments.items():
                            st.write(f"- `{key}`: `{repr(value)}`")
                    else:
                        st.write("_No arguments_")
                else:
                    st.code(str(arguments), language="json")  # Display as code block if not dict
                rendered_indices.add(i)

                # Look ahead for related execution and result steps for *this* tool call
                j = i + 1
                while j < len(steps):
                    next_step = steps[j]
                    # Associate based on sequence and type
                    if next_step.type == "TOOL_EXECUTION":
                        st.write(f"**Status** {WORKFLOW_ICONS['TOOL_EXECUTION']}: {next_step.content}")
                        rendered_indices.add(j)
                    elif next_step.type == "TOOL_RESULT":
                        st.write(f"**Result** {WORKFLOW_ICONS['TOOL_RESULT']}:")
                        details = next_step.details
                        try:
                            # Success, tool execution completed.
                            if isinstance(details, dict):
                                st.json(details)
                            else:
                                details_dict = json.loads(str(details))
                                st.json(details_dict)
                        except (json.JSONDecodeError, TypeError):
                            # Error, tool execution failed.
                            result_str = str(details)
                            st.text(result_str[:500] + ("..." if len(result_str) > 500 else "") or "_Empty result_")
                        rendered_indices.add(j)
                        break  # Stop looking ahead once result is found for this tool
                    elif next_step.type == "TOOL_CALL" or next_step.type == "JSON_TOOL_CALL":
                        # Stop if another tool call starts before finding the result
                        break
                    j += 1

        elif step_type == "JSON_TOOL_CALL":
            # Render LLM-generated tool calls in their own expander
            tool_name = step.details.get("tool_name", "Unknown")
            expander_title = f"{WORKFLOW_ICONS['TOOL_CALL']} LLM Generated Tool Call: {tool_name}"
            with target.expander(expander_title, expanded=False):
                st.write("**Arguments:**")
                arguments = step.details.get("arguments", {})
                if isinstance(arguments, dict):
                    if arguments:
                        for key, value in arguments.items():
                            st.write(f"- `{key}`: `{value}`")
                    else:
                        st.write("_No arguments_")
                else:
                    st.code(str(arguments), language="json")  # Display as code block
            rendered_indices.add(i)

        elif step_type == "ERROR":
            # Display errors directly, outside expanders
            target.error(f"{WORKFLOW_ICONS['ERROR']} {step.content}")
            rendered_indices.add(i)

        # Ignore other step types (USER_QUERY, LLM_THINKING, LLM_RESPONSE, FINAL_STATUS)
        # as they are handled elsewhere (status bar, main message area).


def get_config_hash(config: Configuration, provider: str) -> int:
    """Generate a hash based on relevant configuration settings."""
    relevant_config = {
        "provider": provider,
        "agent_configured": st.session_state.agent_configured,
        "selected_tools": sorted(st.session_state.selected_tools) if st.session_state.agent_configured else None,
        "agent_description": st.session_state.agent_description if st.session_state.agent_configured else None,
        "agent_system_prompt": st.session_state.agent_system_prompt if st.session_state.agent_configured else None,
    }
    if provider == "openai":
        relevant_config.update(
            {
                "model": config._llm_model_name or "",
            }
        )
    elif provider == "anthropic":
        relevant_config.update(
            {
                "model": getattr(config, "_anthropic_model_name", None) or "",
            }
        )
    elif provider == "google":
        relevant_config.update(
            {
                "model": getattr(config, "_google_model_name", None) or "",
            }
        )
    # Hash the sorted representation for consistency
    return hash(json.dumps(relevant_config, sort_keys=True))


async def initialize_mcp_clients(
    config: Configuration, stack: AsyncExitStack, allowed_tools: Optional[set[str]] = None
) -> list[MCPClient]:
    """Initialize MCP Clients based on config.

    Parameters
    ----------
    config : Configuration
        The chatbot configuration
    stack : AsyncExitStack
        Async context manager stack for client lifecycle
    allowed_tools : Optional[set[str]], optional
        If provided, only tools with names in this set will be available

    Returns
    -------
    list[MCPClient]
        List of initialized MCP clients with optionally filtered tools

    """
    clients = []
    server_config_path = ROOT_DIR / "servers_config.json"
    server_config = {}
    if os.path.exists(server_config_path):
        try:
            server_config = config.load_config(str(server_config_path))
        except Exception as e:
            st.warning(f"Failed to load MCP server config for client init: {e}")

    if server_config and "mcpServers" in server_config:
        for name, srv_config in server_config["mcpServers"].items():
            try:
                client = MCPClient(name, srv_config)
                # Enter the client's context into the provided stack
                await stack.enter_async_context(client)

                # Filter tools if allowed_tools is specified
                if allowed_tools is not None:
                    client.filter_tools(allowed_tools)

                clients.append(client)
            except Exception as client_ex:
                st.error(f"Failed to initialize MCP client {name}: {client_ex}")
    return clients


async def process_chat(user_input: str) -> None:
    """Handle user input, interacts with the backend."""
    # 1. Add user message to state and display it
    # Use a copy for history to avoid potential modification issues if session resets
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Prepare for assistant response
    current_workflow_steps = []
    final_status_message = "Processing finished."  # Initialize with default value

    # Create a placeholder for the assistant response that will be shown during processing
    assistant_container = st.container()
    with assistant_container, st.chat_message("assistant"):
        status_placeholder = st.status("Processing...", expanded=False)
        workflow_display_container = st.empty()
        message_placeholder = st.empty()

    try:
        # Session and Client Management
        config = st.session_state.chatbot_config
        current_config_hash = get_config_hash(config, st.session_state.llm_provider)

        # Check if config changed or session doesn't exist
        if st.session_state.chat_session is None or current_config_hash != st.session_state.session_config_hash:
            # st.toast(
            # "Configuration changed or first run, initializing new chat session."
            # )
            # If config changed, clear previous messages and reset state
            if (
                st.session_state.session_config_hash is not None
                and current_config_hash != st.session_state.session_config_hash
            ):
                st.session_state.messages = [{"role": "user", "content": user_input}]  # Keep only current input
                # Need to properly exit previous stack if it exists
                if st.session_state.mcp_client_stack:
                    await st.session_state.mcp_client_stack.__aexit__(None, None, None)
                    st.session_state.mcp_client_stack = None
                    st.session_state.active_mcp_clients = []

            # Create LLM Client
            try:
                llm_client = create_llm_client(provider=st.session_state.llm_provider, config=config)
            except NotImplementedError:
                st.error(
                    f"❌ {st.session_state.llm_provider.title()} provider is not yet implemented. Please use OpenAI for now."
                )
                return
            except Exception as e:
                st.error(f"❌ Failed to create LLM client: {e}")
                return

            if not llm_client:
                raise ValueError("LLM Client could not be created. Check configuration.")

            # Create and manage MCP Clients using
            # an AsyncExitStack stored in session state
            st.session_state.mcp_client_stack = AsyncExitStack()
            # Use selected tools if agent is configured, otherwise use all tools
            allowed_tools = st.session_state.selected_tools if st.session_state.agent_configured else None
            mcp_clients = await initialize_mcp_clients(config, st.session_state.mcp_client_stack, allowed_tools)
            st.session_state.active_mcp_clients = mcp_clients  # Store references

            # Create new ChatSession
            # Pass the *active* clients.
            # ChatSession needs to handle these potentially changing.
            # Assuming ChatSession uses the clients passed at creation time.
            custom_prompt = (
                st.session_state.agent_system_prompt
                if st.session_state.agent_configured and st.session_state.agent_system_prompt.strip()
                else None
            )
            st.session_state.chat_session = ChatSession(st.session_state.active_mcp_clients, llm_client, custom_prompt)
            await st.session_state.chat_session.initialize()
            # Keep the history messages from the new chat session.
            if not st.session_state.history_messages:
                # If the history messages are not set, we need to get the
                # system prompt from the chat session.
                st.session_state.history_messages = st.session_state.chat_session.messages
            st.session_state.session_config_hash = current_config_hash
            # st.toast("New chat session initialized.", icon="🎈")  # User feedback
        else:
            # Ensure clients are available if session exists
            # (they should be in active_mcp_clients)
            # This part assumes the clients associated with the existing session
            # are still valid. If tool refresh happens, this might need adjustment.
            pass  # Use existing session

        if not st.session_state.chat_session:
            raise RuntimeError("Chat session could not be initialized.")

        chat_session = st.session_state.chat_session
        chat_session.messages = st.session_state.history_messages
        print("Chat session messages:", chat_session.messages)

        # Add user query to workflow steps
        current_workflow_steps.append(WorkflowStep(type="USER_QUERY", content=user_input))
        with workflow_display_container.container():
            render_workflow([], container=st)  # Render empty initially

        tool_call_count = 0
        active_tool_name = None
        mcp_tool_calls_made = False

        # Initial thinking step (not added to workflow steps for display here)
        status_placeholder.update(label="🧠 Processing request...", state="running", expanded=False)

        # Stream response handling
        accumulated_response_content = ""  # Accumulate raw response content
        new_step_added = False  # Track if workflow needs rerender

        # Process streaming response using the persistent chat_session
        print("Now chat session messages:", chat_session.messages)
        async for result in chat_session.send_message_stream(user_input, show_workflow=True):
            new_step_added = False  # Reset for this iteration
            if isinstance(result, tuple):
                status, content = result

                if status == "status":
                    status_placeholder.update(label=f"🧠 {content}", state="running")
                elif status == "tool_call":
                    mcp_tool_calls_made = True
                    tool_call_count += 1
                    active_tool_name = content
                    tool_call_step = WorkflowStep(
                        type="TOOL_CALL",
                        content=f"Initiating call to: {content}",
                        details={"tool_name": content, "arguments": "Pending..."},
                    )
                    current_workflow_steps.append(tool_call_step)
                    new_step_added = True
                    status_placeholder.update(label=f"🔧 Calling tool: {content}", state="running")
                elif status == "tool_arguments":
                    if active_tool_name:
                        updated = False
                        for step in reversed(current_workflow_steps):
                            if (
                                step.type == "TOOL_CALL"
                                and step.details.get("tool_name") == active_tool_name
                                and step.details.get("arguments") == "Pending..."
                            ):
                                try:
                                    step.details["arguments"] = json.loads(content)
                                except json.JSONDecodeError:
                                    step.details["arguments"] = content
                                updated = True
                                break
                        if updated:
                            new_step_added = True
                elif status == "tool_execution":
                    current_workflow_steps.append(WorkflowStep(type="TOOL_EXECUTION", content=content))
                    new_step_added = True
                    status_placeholder.update(label=f"⚡ {content}", state="running")
                elif status == "tool_results":
                    current_workflow_steps.append(
                        WorkflowStep(
                            type="TOOL_RESULT",
                            content="Received result.",
                            details=content,
                        )
                    )
                    new_step_added = True
                    status_placeholder.update(
                        label=f"🧠 Processing results from {active_tool_name}...",
                        state="running",
                    )
                    active_tool_name = None

                elif status == "response":
                    if isinstance(content, str):
                        accumulated_response_content += content
                        potential_json_tools, clean_response_so_far = extract_json_tool_calls(
                            accumulated_response_content
                        )
                        message_placeholder.markdown(clean_response_so_far + "▌")
                        status_placeholder.update(label="💬 Streaming response...", state="running")

                elif status == "error":
                    error_content = str(content)
                    error_step = WorkflowStep(type="ERROR", content=error_content)
                    current_workflow_steps.append(error_step)
                    new_step_added = True
                    status_placeholder.update(
                        label=f"❌ Error: {error_content[:100]}...",
                        state="error",
                        expanded=True,
                    )
                    message_placeholder.error(f"An error occurred: {error_content}")
                    with workflow_display_container.container():
                        render_workflow(current_workflow_steps, container=st)
                    break  # Stop processing on error

            else:  # Handle non-tuple results (e.g., direct string) if necessary
                if isinstance(result, str):
                    accumulated_response_content += result
                    potential_json_tools, clean_response_so_far = extract_json_tool_calls(accumulated_response_content)
                    message_placeholder.markdown(clean_response_so_far + "▌")
                    status_placeholder.update(label="💬 Streaming response...", state="running")

            # Re-render the workflow area if a new step was added
            if new_step_added:
                with workflow_display_container.container():
                    render_workflow(current_workflow_steps, container=st)

        # 3. Post-stream processing and final display

        json_tools, clean_response = extract_json_tool_calls(accumulated_response_content)
        final_display_content = clean_response.strip()

        json_tools_added = False
        for json_tool in json_tools:
            if not mcp_tool_calls_made:  # Heuristic: only add if no standard calls
                tool_name = json_tool.get("tool", "unknown_tool")
                tool_args = json_tool.get("arguments", {})
                json_step = WorkflowStep(
                    type="JSON_TOOL_CALL",
                    content=f"LLM generated tool call: {tool_name}",
                    details={"tool_name": tool_name, "arguments": tool_args},
                )
                current_workflow_steps.append(json_step)
                tool_call_count += 1
                json_tools_added = True

        if not final_display_content and json_tools_added:
            final_display_content = ""  # Or a message like "Generated tool calls."

        message_placeholder.markdown(final_display_content or "_No text response_")

        if final_display_content:
            llm_response_step = WorkflowStep(
                type="LLM_RESPONSE",
                content="Final response generated.",
                details={"response_text": final_display_content},
            )
            current_workflow_steps.append(llm_response_step)

        final_status_message = "Completed."
        if tool_call_count > 0:
            final_status_message += f" Processed {tool_call_count} tool call(s)."
        current_workflow_steps.append(WorkflowStep(type="FINAL_STATUS", content=final_status_message))

        status_placeholder.update(label=f"✅ {final_status_message}", state="complete", expanded=False)

        with workflow_display_container.container():
            render_workflow(current_workflow_steps, container=st)

        # --- Store results in session state ---
        # Store the assistant response for the next rerun
        assistant_message = {
            "role": "assistant",
            "content": final_display_content or accumulated_response_content,  # Store clean or full
            "workflow_steps": [step.to_dict() for step in current_workflow_steps],
        }
        st.session_state.messages.append(assistant_message)

        # Clear the temporary assistant container and rerun to show final state
        assistant_container.empty()
        st.rerun()
        # --- End storing results ---

    except Exception as e:
        error_message = f"An unexpected error occurred in process_chat: {str(e)}"
        st.error(error_message)
        current_workflow_steps.append(WorkflowStep(type="ERROR", content=error_message))
        try:
            with workflow_display_container.container():
                render_workflow(current_workflow_steps, container=st)
        except Exception as render_e:
            st.error(f"Additionally, failed to render workflow after error: {render_e}")

        status_placeholder.update(label=f"❌ Error: {error_message[:100]}...", state="error", expanded=True)
        # Append error message to history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Error: {error_message}",
                "workflow_steps": [step.to_dict() for step in current_workflow_steps],
            }
        )
        # Clear the temporary assistant container and rerun to show final state
        assistant_container.empty()
        st.rerun()
    finally:
        # --- Final UI update ---
        if status_placeholder._label != f"✅ {final_status_message}" and status_placeholder._state != "error":
            status_placeholder.update(label="Processing finished.", state="complete", expanded=False)

        # ------------------------------------------------------------------
        # IMPORTANT CLEAN‑UP!
        #
        # Each Streamlit rerun executes this script in a *fresh* asyncio
        # event‑loop.  Any MCPClient / ChatSession objects created in a
        # previous loop become invalid and will raise
        # "Attempted to exit cancel scope in a different task…" errors when
        # they try to close themselves later on.
        #
        # Therefore we:
        #   1. Close the AsyncExitStack that owns all MCP clients *inside the
        #      same loop that created them* (`process_chat`'s loop).
        #   2. Drop the references from `st.session_state` so a new set of
        #      clients / ChatSession are created on the next user message.
        # ------------------------------------------------------------------
        try:
            if st.session_state.mcp_client_stack is not None:
                await st.session_state.mcp_client_stack.__aexit__(None, None, None)
        except Exception as cleanup_exc:
            # Log but do not crash UI – the loop is ending anyway.
            print("MCP clean‑up error:", cleanup_exc, file=sys.stderr)
        finally:
            st.session_state.mcp_client_stack = None
            st.session_state.active_mcp_clients = []
            # Do *not* reuse async objects across Streamlit reruns.
            st.session_state.history_messages = chat_session.messages
            st.session_state.chat_session = None


def display_chat_history() -> None:
    """Displays the chat history from st.session_state.messages."""
    for idx, message in enumerate(st.session_state.messages):
        # Use a unique key for each chat message element
        with st.chat_message(message["role"]):
            # Workflow Rendering
            if message["role"] == "assistant" and "workflow_steps" in message:
                # Use a unique key for the workflow container
                workflow_history_container = st.container()

                workflow_steps = []
                if isinstance(message["workflow_steps"], list):
                    for step_dict in message["workflow_steps"]:
                        if isinstance(step_dict, dict):
                            workflow_steps.append(
                                WorkflowStep(
                                    type=step_dict.get("type", "UNKNOWN"),
                                    content=step_dict.get("content", ""),
                                    details=step_dict.get("details", {}),
                                )
                            )
                if workflow_steps:
                    render_workflow(workflow_steps, container=workflow_history_container)

            # Message Content Rendering (Rendered after workflow for assistant)
            # Allow basic HTML if needed
            st.markdown(message["content"], unsafe_allow_html=True)


def render_agent_configuration(mcp_tools: dict[str, list[MCPTool]]) -> bool:
    """Render agent configuration screen and return True if agent is configured."""
    st.header("🔧 Configure Your Declarative Agent")
    st.write("Define your AI assistant by selecting a name, model, and tools.")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📋 Basic Configuration")

            # Agent name
            agent_name = st.text_input(
                "Agent Name:",
                value=st.session_state.agent_name,
                placeholder="e.g., FileManager Assistant, Data Analyzer",
                help="Give your agent a descriptive name",
            )

            # Agent description
            agent_description = st.text_area(
                "Agent Description:",
                value=st.session_state.agent_description,
                placeholder="e.g., A specialized assistant for managing files and folders, helping with data analysis and visualization tasks.",
                help="Describe what your agent is designed to do",
                height=80,
            )

            # Provider selection
            available_providers = list(get_args(LLMProvider))
            provider_index = 0
            if st.session_state.agent_provider in available_providers:
                provider_index = available_providers.index(st.session_state.agent_provider)

            agent_provider = st.selectbox(
                "LLM Provider:", options=available_providers, index=provider_index, help="Choose the AI model provider"
            )

            # Model selection based on provider
            agent_model = ""
            if agent_provider == "openai":
                openai_models = ["gpt-4o", "gpt-4o-mini"]
                model_index = 0
                if st.session_state.agent_model in openai_models:
                    model_index = openai_models.index(st.session_state.agent_model)
                agent_model = st.selectbox("Model:", options=openai_models, index=model_index)
            elif agent_provider == "anthropic":
                st.info("ℹ️ Anthropic support is coming soon!")
                anthropic_models = ["claude-4-sonnet", "claude-4-opus", "claude-3.7-sonnet"]
                agent_model = st.selectbox("Model:", options=anthropic_models, index=0, disabled=True)
            elif agent_provider == "google":
                st.info("ℹ️ Google support is coming soon!")
                google_models = ["gemini-2.5-pro", "gemini-2.5-flash"]
                agent_model = st.selectbox("Model:", options=google_models, index=0, disabled=True)

            # System prompt configuration
            st.subheader("🎯 Custom System Prompt (Optional)")
            agent_system_prompt = st.text_area(
                "System Prompt:",
                value=st.session_state.agent_system_prompt,
                placeholder="You're a helpful assistant who speaks heavy British slang.",
                help="Optional: Define how your agent should behave and respond. Tool usage instructions will be automatically appended.",
                height=120,
            )

        with col2:
            st.subheader("🛠️ Tool Selection")

            if not mcp_tools:
                st.warning("No MCP tools available. Please check your server configuration.")
                st.button("🔄 Refresh Tools", on_click=lambda: st.session_state.update({"mcp_tools_cache": {}}))
                return False

            st.write("Select the tools your agent can use:")

            # Display tools by client/server
            selected_tools = set(st.session_state.selected_tools)

            for client_name, client_tools in mcp_tools.items():
                with st.expander(f"📦 {client_name} ({len(client_tools)} tools)", expanded=True):
                    if not client_tools:
                        st.write("No tools available in this server.")
                        continue

                    # Select all/none buttons for this client
                    col_all, col_none = st.columns(2)
                    with col_all:
                        if st.button("Select All", key=f"select_all_{client_name}"):
                            for tool in client_tools:
                                selected_tools.add(tool.name)
                    with col_none:
                        if st.button("Select None", key=f"select_none_{client_name}"):
                            for tool in client_tools:
                                selected_tools.discard(tool.name)

                    # Individual tool checkboxes
                    for tool in client_tools:
                        col_checkbox, col_schema = st.columns([3, 1])

                        with col_checkbox:
                            is_selected = st.checkbox(
                                f"**{tool.name}**",
                                value=tool.name in selected_tools,
                                key=f"tool_{client_name}_{tool.name}",
                                help=tool.description,
                            )
                            if is_selected:
                                selected_tools.add(tool.name)
                            else:
                                selected_tools.discard(tool.name)

                        with col_schema, st.popover("Schema", use_container_width=True):
                            st.json(tool.input_schema)

    # Configuration summary
    st.subheader("📊 Configuration Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Agent Name", agent_name or "Not set")
    with col2:
        st.metric("Model", f"{agent_provider}/{agent_model}" if agent_model else "Not set")
    with col3:
        st.metric("Selected Tools", len(selected_tools))
    with col4:
        st.metric("Custom Prompt", "Yes" if agent_system_prompt.strip() else "Default")

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        can_configure = bool(agent_name.strip() and agent_model and selected_tools)

        if st.button("✅ Configure Agent", type="primary", disabled=not can_configure, use_container_width=True):
            # Save configuration
            st.session_state.agent_name = agent_name.strip()
            st.session_state.agent_description = agent_description.strip()
            st.session_state.agent_system_prompt = agent_system_prompt.strip()
            st.session_state.agent_provider = agent_provider
            st.session_state.agent_model = agent_model
            st.session_state.selected_tools = selected_tools
            st.session_state.agent_configured = True

            # Update the main configuration for consistency
            st.session_state.llm_provider = agent_provider
            if agent_provider == "openai":
                st.session_state.chatbot_config._llm_model_name = agent_model

            # Reset chat state to use new configuration
            st.session_state.messages = []
            st.session_state.chat_session = None
            st.session_state.session_config_hash = None
            st.session_state.active_mcp_clients = []
            st.session_state.mcp_client_stack = None
            st.session_state.history_messages = []

            st.success(f"✅ Agent '{agent_name}' configured successfully!")
            st.balloons()
            st.rerun()

    with col3:
        if st.button("🔄 Reset Configuration", use_container_width=True):
            st.session_state.agent_configured = False
            st.session_state.agent_name = ""
            st.session_state.agent_description = ""
            st.session_state.agent_system_prompt = ""
            st.session_state.agent_model = ""
            st.session_state.agent_provider = "openai"
            st.session_state.selected_tools = set()
            st.rerun()

    if not can_configure:
        st.warning(
            "⚠️ Please provide an agent name, select a model, and choose at least one tool to configure your agent."
        )

    return False


async def main() -> None:
    """Main application entry point."""
    # Get MCP tools (cached) - Tool list displayed in sidebar or configuration
    mcp_tools = await get_mcp_tools()

    # Check if agent is configured
    if not st.session_state.agent_configured:
        # Show agent configuration screen
        render_agent_configuration(mcp_tools)
        return

    # Show agent reconfiguration option in sidebar
    with st.sidebar:
        st.header("🤖 Current Declarative Agent")
        st.write(f"**Name:** {st.session_state.agent_name}")
        if st.session_state.agent_description:
            st.write(f"**Description:** {st.session_state.agent_description}")
        st.write(f"**Model:** {st.session_state.agent_provider}/{st.session_state.agent_model}")
        # Show selected tools
        with st.expander("🛠️ Selected Tools", expanded=False):
            if st.session_state.selected_tools:
                for tool_name in sorted(st.session_state.selected_tools):
                    st.write(f"• {tool_name}")
            else:
                st.write("No tools selected")
        if st.session_state.agent_system_prompt:
            with st.expander("🎯 Custom System Prompt", expanded=False):
                st.write(st.session_state.agent_system_prompt)

        if st.button("🔧 Reconfigure Agent", use_container_width=True, type="primary"):
            st.session_state.agent_configured = False
            st.session_state.messages = []
            st.session_state.chat_session = None
            st.session_state.session_config_hash = None
            st.session_state.active_mcp_clients = []
            st.session_state.mcp_client_stack = None
            st.session_state.history_messages = []
            st.rerun()

        # Clear chat button
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_session = None
            st.session_state.session_config_hash = None
            st.session_state.active_mcp_clients = []
            st.session_state.mcp_client_stack = None
            st.session_state.history_messages = []
            st.toast("Chat cleared!", icon="🧹")
            st.rerun()

    # Display existing chat messages and their workflows from session state
    display_chat_history()

    # Handle new chat input
    if prompt := st.chat_input(f"Ask {st.session_state.agent_name} something..."):
        await process_chat(prompt)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        # Note: Reliable async cleanup on shutdown is still complex in Streamlit
        pass
