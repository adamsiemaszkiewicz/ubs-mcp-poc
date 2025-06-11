import os
from contextlib import AsyncExitStack

import streamlit as st

from src.consts import ROOT_DIR
from src.mcp.client import MCPClient


async def get_mcp_tools(force_refresh: bool = False) -> dict[str, dict]:
    """Get MCP tools from cache or by initializing clients.

    Parameters
    ----------
    force_refresh : bool, optional
        Whether to force refresh the tools cache, by default False

    Returns
    -------
    dict[str, dict]
        Dictionary mapping server names to server info containing:
        - 'tools': list[MCPTool] - Available tools from the server
        - 'transport': str - Transport type ('HTTP', 'STDIO')
        - 'config': dict - Server configuration details

    """
    if not force_refresh and st.session_state.mcp_tools_cache:
        return st.session_state.mcp_tools_cache

    tools_dict = {}
    config = st.session_state.chatbot_config
    server_config_path = ROOT_DIR / "servers_config.json"

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

                # Determine transport type
                transport_type = "HTTP" if "url" in srv_config else "STDIO"

                tools_dict[name] = {"tools": tools, "transport": transport_type, "config": srv_config}
            except Exception as e:
                st.sidebar.error(f"Error fetching tools from {name}: {e}")

    st.session_state.mcp_tools_cache = tools_dict
    return tools_dict
