import json
import re
import sys
from contextlib import AsyncExitStack
from typing import Any, Optional

import streamlit as st

from src.config import Configuration
from src.consts import ROOT_DIR
from src.mcp.client import MCPClient
from src.model.factory import create_llm_client
from src.ui.components import WorkflowStep


def extract_json_tool_calls(text: str) -> tuple[list[dict[str, Any]], str]:
    """Extract tool call JSON objects from text using robust pattern matching.

    Uses similar logic to ChatSession._extract_tool_calls but adapted for our needs.

    Parameters
    ----------
    text : str
        Text possibly containing JSON tool calls

    Returns
    -------
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
        for start, end in sorted(extracted_indices):
            cleaned_parts.append(text[last_end:start])
            last_end = end
        cleaned_parts.append(text[last_end:])
        cleaned_text = "".join(cleaned_parts).strip()
    else:
        # If no JSON tool calls were extracted via regex,
        # the original text is the cleaned text
        cleaned_text = text

    return tool_calls, cleaned_text


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
    allowed_tools : Optional[Set[str]], optional
        If provided, only tools with names in this set will be available

    Returns
    -------
    List[MCPClient]
        List of initialized MCP clients with optionally filtered tools

    """
    import os

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
    """Handle user input, interacts with the backend.

    This function coordinates the entire chat processing workflow including
    session management, tool calls, and response streaming.
    """
    from src.mcp.session import ChatSession

    # 1. Add user message to state and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Prepare for assistant response
    current_workflow_steps = []
    final_status_message = "Processing finished."

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

            # Create and manage MCP Clients using an AsyncExitStack stored in session state
            st.session_state.mcp_client_stack = AsyncExitStack()
            # Use selected tools if agent is configured, otherwise use all tools
            allowed_tools = st.session_state.selected_tools if st.session_state.agent_configured else None
            mcp_clients = await initialize_mcp_clients(config, st.session_state.mcp_client_stack, allowed_tools)
            st.session_state.active_mcp_clients = mcp_clients

            # Create new ChatSession
            custom_prompt = (
                st.session_state.agent_system_prompt
                if st.session_state.agent_configured and st.session_state.agent_system_prompt.strip()
                else None
            )
            st.session_state.chat_session = ChatSession(st.session_state.active_mcp_clients, llm_client, custom_prompt)
            await st.session_state.chat_session.initialize()
            # Keep the history messages from the new chat session.
            if not st.session_state.history_messages:
                st.session_state.history_messages = st.session_state.chat_session.messages
            st.session_state.session_config_hash = current_config_hash

        if not st.session_state.chat_session:
            raise RuntimeError("Chat session could not be initialized.")

        chat_session = st.session_state.chat_session
        chat_session.messages = st.session_state.history_messages

        # Add user query to workflow steps
        current_workflow_steps.append(WorkflowStep(type="USER_QUERY", content=user_input))

        tool_call_count = 0
        active_tool_name = None
        mcp_tool_calls_made = False

        # Initial thinking step
        status_placeholder.update(label="🧠 Processing request...", state="running", expanded=False)

        # Stream response handling
        accumulated_response_content = ""
        new_step_added = False

        # Process streaming response using the persistent chat_session
        async for result in chat_session.send_message_stream(user_input, show_workflow=True):
            new_step_added = False
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
                    # Parse content if it's a JSON string, otherwise use as-is
                    try:
                        if isinstance(content, str):
                            parsed_details = json.loads(content)
                        else:
                            parsed_details = content if isinstance(content, dict) else {"result": content}
                    except json.JSONDecodeError:
                        parsed_details = {"result": content}

                    current_workflow_steps.append(
                        WorkflowStep(
                            type="TOOL_RESULT",
                            content="Received result.",
                            details=parsed_details,
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
                    # Import here to avoid circular imports
                    from src.ui.components import render_workflow

                    with workflow_display_container.container():
                        render_workflow(current_workflow_steps, container=st)
                    break

            else:  # Handle non-tuple results if necessary
                if isinstance(result, str):
                    accumulated_response_content += result
                    potential_json_tools, clean_response_so_far = extract_json_tool_calls(accumulated_response_content)
                    message_placeholder.markdown(clean_response_so_far + "▌")
                    status_placeholder.update(label="💬 Streaming response...", state="running")

            # Re-render the workflow area if a new step was added
            if new_step_added:
                from src.ui.components import render_workflow

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
            final_display_content = ""

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

        from src.ui.components import render_workflow

        with workflow_display_container.container():
            render_workflow(current_workflow_steps, container=st)

        # --- Store results in session state ---
        assistant_message = {
            "role": "assistant",
            "content": final_display_content or accumulated_response_content,
            "workflow_steps": [step.model_dump() for step in current_workflow_steps],
        }
        st.session_state.messages.append(assistant_message)

        # Clear the temporary assistant container and rerun to show final state
        assistant_container.empty()
        st.rerun()

    except Exception as e:
        error_message = f"An unexpected error occurred in process_chat: {str(e)}"
        st.error(error_message)
        current_workflow_steps.append(WorkflowStep(type="ERROR", content=error_message))
        try:
            from src.ui.components import render_workflow

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
                "workflow_steps": [step.model_dump() for step in current_workflow_steps],
            }
        )
        # Clear the temporary assistant container and rerun to show final state
        assistant_container.empty()
        st.rerun()
    finally:
        # --- Final UI update ---
        if status_placeholder._label != f"✅ {final_status_message}" and status_placeholder._state != "error":
            status_placeholder.update(label="Processing finished.", state="complete", expanded=False)

        # Clean-up async resources
        try:
            if st.session_state.mcp_client_stack is not None:
                await st.session_state.mcp_client_stack.__aexit__(None, None, None)
        except Exception as cleanup_exc:
            print("MCP clean‑up error:", cleanup_exc, file=sys.stderr)
        finally:
            st.session_state.mcp_client_stack = None
            st.session_state.active_mcp_clients = []
            # Store history messages but reset session for next interaction
            st.session_state.history_messages = chat_session.messages
            st.session_state.chat_session = None
