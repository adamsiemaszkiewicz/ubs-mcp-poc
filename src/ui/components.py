from typing import Any

import streamlit as st
from pydantic import BaseModel, Field


class WorkflowStep(BaseModel):
    """Workflow step class for tracking chatbot interactions in Streamlit UI."""

    type: str
    content: str
    details: dict[str, Any] = Field(default_factory=dict)


# --- Constants for Streamlit UI ---
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


def render_workflow(steps: list[WorkflowStep], container: Any = None) -> None:
    """Render workflow steps in Streamlit, placing each tool call sequence in its own expander.

    Parameters
    ----------
    steps : list[WorkflowStep]
        List of workflow steps to render
    container : Any, optional
        Streamlit container to render in, by default None (uses main area)

    """
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
                    st.code(str(arguments), language="json")
                rendered_indices.add(i)

                # Look ahead for related execution and result steps
                j = i + 1
                while j < len(steps):
                    next_step = steps[j]
                    if next_step.type == "TOOL_EXECUTION":
                        st.write(f"**Status** {WORKFLOW_ICONS['TOOL_EXECUTION']}: {next_step.content}")
                        rendered_indices.add(j)
                    elif next_step.type == "TOOL_RESULT":
                        st.write(f"**Result** {WORKFLOW_ICONS['TOOL_RESULT']}:")
                        details = next_step.details
                        try:
                            if isinstance(details, dict):
                                st.json(details)
                            else:
                                import json

                                details_dict = json.loads(str(details))
                                st.json(details_dict)
                        except (json.JSONDecodeError, TypeError):
                            result_str = str(details)
                            st.text(result_str[:500] + ("..." if len(result_str) > 500 else "") or "_Empty result_")
                        rendered_indices.add(j)
                        break
                    elif next_step.type == "TOOL_CALL" or next_step.type == "JSON_TOOL_CALL":
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
                    st.code(str(arguments), language="json")
            rendered_indices.add(i)

        elif step_type == "ERROR":
            # Display errors directly, outside expanders
            target.error(f"{WORKFLOW_ICONS['ERROR']} {step.content}")
            rendered_indices.add(i)


def display_chat_history() -> None:
    """Display the chat history from st.session_state.messages."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Workflow Rendering
            if message["role"] == "assistant" and "workflow_steps" in message:
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

            # Message Content Rendering
            st.markdown(message["content"], unsafe_allow_html=True)


def render_agent_configuration(mcp_tools: dict[str, dict]) -> bool:
    """Render agent configuration screen and return True if agent is configured."""
    from typing import get_args

    from src.model.factory import LLMProvider

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
                openai_models = ["gpt-4o-mini","gpt-4o"]
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
                placeholder="Define how your agent should behave and respond. Tool usage instructions will be automatically appended.",
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

            for client_name, server_info in mcp_tools.items():
                client_tools = server_info["tools"]
                transport_type = server_info["transport"]
                config = server_info["config"]

                # Create transport type icon and description
                transport_icon = "🌐" if transport_type == "HTTP" else "🔗"
                transport_info = ""
                if transport_type == "HTTP":
                    transport_info = f" (HTTP: {config.get('url', 'Unknown URL')})"
                else:
                    command = config.get("command", "Unknown")
                    transport_info = f" (STDIO: {command})"

                with st.expander(
                    f"{transport_icon} {client_name} ({len(client_tools)} tools){transport_info}", expanded=False
                ):
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


def render_sidebar() -> None:
    """Render the sidebar with agent information and controls."""
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
