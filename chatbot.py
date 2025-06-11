import asyncio
import sys

import streamlit as st

from src.chat.processor import process_chat
from src.mcp.manager import get_mcp_tools
from src.ui.components import display_chat_history, render_agent_configuration, render_sidebar
from src.ui.manager import initialize_session_state

# Platform-specific event loop policy for Windows
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="MCP Declarative Agent", layout="wide")
st.title("🤖 MCP Declarative Agent")
st.caption("Configure and chat with a Declarative Agent capable of using MCP tools.")

# --- Initialize Session State ---
initialize_session_state()


async def main() -> None:
    """Run Declarative Agentchatbot."""
    # Get MCP tools (cached) - Tool list displayed in sidebar or configuration
    mcp_tools = await get_mcp_tools()

    # Check if agent is configured
    if not st.session_state.agent_configured:
        # Show agent configuration screen
        render_agent_configuration(mcp_tools)
        return

    # Show agent information and controls in sidebar
    render_sidebar()

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
