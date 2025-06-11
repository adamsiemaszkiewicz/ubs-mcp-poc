import streamlit as st

from src.config import Configuration


def initialize_session_state() -> None:
    """Initialize all session state variables with their default values."""
    # Core message history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # LLM Provider configuration
    if "llm_provider" not in st.session_state:
        st.session_state.llm_provider = "openai"

    # Main chatbot configuration
    if "chatbot_config" not in st.session_state:
        st.session_state.chatbot_config = Configuration()

    # MCP tools cache
    if "mcp_tools_cache" not in st.session_state:
        st.session_state.mcp_tools_cache = {}

    # Declarative agent configuration state
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

    # Chat session and config tracking
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


def reset_chat_state() -> None:
    """Reset chat-related session state while preserving agent configuration."""
    st.session_state.messages = []
    st.session_state.chat_session = None
    st.session_state.session_config_hash = None
    st.session_state.active_mcp_clients = []
    st.session_state.mcp_client_stack = None
    st.session_state.history_messages = []


def reset_agent_configuration() -> None:
    """Reset all agent configuration to defaults."""
    st.session_state.agent_configured = False
    st.session_state.agent_name = ""
    st.session_state.agent_description = ""
    st.session_state.agent_system_prompt = ""
    st.session_state.agent_model = ""
    st.session_state.agent_provider = "openai"
    st.session_state.selected_tools = set()
    # Also reset chat state when configuration changes
    reset_chat_state()


def get_agent_summary() -> dict:
    """Get a summary of the current agent configuration.

    Returns
    -------
    dict
        Dictionary containing agent configuration summary

    """
    return {
        "configured": st.session_state.agent_configured,
        "name": st.session_state.agent_name,
        "description": st.session_state.agent_description,
        "provider": st.session_state.agent_provider,
        "model": st.session_state.agent_model,
        "tools_count": len(st.session_state.selected_tools),
        "has_custom_prompt": bool(st.session_state.agent_system_prompt.strip()),
    }
