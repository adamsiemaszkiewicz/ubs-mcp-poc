# Model Context Protocol Chatbot

## 🌟 Overview

This project demonstrates a production-ready MCP client implementation that acts as an interactive chatbot interface. The Model Context Protocol (MCP) is an open standard that provides a unified way for AI applications to securely connect to diverse data sources and tools, similar to how USB-C provides standardized connectivity for devices.

### What is Model Context Protocol?

The **Model Context Protocol (MCP)** is an open protocol that standardizes how applications provide context to LLMs. It follows a client-server architecture where:

- **MCP Host**: Application like this chatbot that contain MCP clients and coordinate AI integration
- **MCP Client**: Protocol client that maintain 1:1 connections with servers  
- **MCP Servers**: Lightweight programs exposing specific capabilities through standardized interfaces

```mermaid
flowchart LR
    subgraph "Chatbot Application (Host)"
        H[Streamlit Interface]
        C1[MCP Client 1]
        C2[MCP Client 2]
        C3[MCP Client 3]
        H --> C1
        H --> C2  
        H --> C3
    end

    subgraph "Local Machine"
        S1[Markdown Server]
        S2[DateTime Server]
        R1[("Local Files")]
        R2[("System Data")]
        
        C1 --> S1
        C2 --> S2
        S1 <--> R1
        S2 <--> R2
    end

    subgraph "External"
        S3[Filesystem Server]
        R3[("File System")]
        
        C3 --> S3
        S3 <--> R3
    end
```

## 🏗️ Architecture

### Core Components

This project implements the complete MCP architecture with the following key components:

#### 1. **Host Application** (`chatbot.py`)
The main Streamlit application that acts as the MCP host:
- **Role**: Coordinates MCP client and manages the overall UX/UI
- **Responsibilities**: 
  - User interface and chat management
  - LLM integration and response streaming
  - Client lifecycle management
  - Security and authorization control
  - Context aggregation across multiple servers

#### 2. **MCP Client Layer** (`src/mcp/client.py`)

**🔗 Core Responsibility: One Client ↔ One Server Communication**

Each `MCPClient` manages a single connection to one MCP server, following the official 1:1 relationship pattern. The client automatically detects and supports both **STDIO** and **HTTP** transports based on configuration.

##### **🏗️ Basic Architecture Flow**

```mermaid
sequenceDiagram
    participant App as ChatBot App
    participant Client as MCPClient
    participant Server as MCP Server
    
    App->>Client: Initialize connection
    Client->>Server: STDIO/HTTP connection + handshake
    Server-->>Client: Capabilities + available tools
    Client-->>App: Ready with tool list
    
    App->>Client: Execute tool("read_file", {path: "..."})
    Client->>Server: MCP tools/call message
    Server-->>Client: Tool result
    Client-->>App: Formatted result
```

##### **💡 Core Implementation Logic**

**Step 1: Connection Setup (Auto-Transport Detection)**
```python
class MCPClient:
    async def initialize(self):
        # 1. Auto-detect transport type from config
        if "url" in self.config:
            # HTTP transport - connect to running server
            transport = await streamablehttp_client(
                url=self.config["url"],
                headers=self.config.get("headers"),
                timeout=timedelta(seconds=self.config.get("timeout", 30))
            )
            read, write, get_session_id = transport
        else:
            # STDIO transport - start server process
            server_process = start_server_process(self.config)
            transport = await stdio_client(server_process)
            read, write = transport
        
        # 2. MCP handshake - exchange protocol versions
        session = ClientSession(read, write)
        await session.initialize()  # ← MCP protocol negotiation
        
        # 3. Discover what tools this server provides
        tools_response = await session.list_tools()
        self.available_tools = parse_tools(tools_response)
```

**Step 2: Tool Execution**
```python
async def execute_tool(self, tool_name, arguments):
    # Validate tool exists
    if tool_name not in self.available_tools:
        raise ValueError(f"Tool {tool_name} not available")
    
    # Execute with retry logic
    for attempt in range(retries):
        try:
            # Send MCP tools/call request
            result = await self.session.call_tool(tool_name, arguments)
            return result
        except Exception as e:
            if attempt < retries - 1:
                await asyncio.sleep(backoff_delay)  # Retry with backoff
            else:
                raise e
```

##### **🎯 Key MCP Concepts in Action**

**Tool Discovery Pattern**
```python
# Each client discovers its server's tools independently
async def discover_tools(self):
    """What can this specific server do?"""
    response = await self.session.list_tools()
    
    # Example response processing:
    # [{name: "read_file", description: "...", schema: {...}},
    #  {name: "write_file", description: "...", schema: {...}}]
    
    self.tools = [MCPTool(name=t.name, 
                         description=t.description,
                         schema=t.inputSchema) 
                 for t in response.tools]
```

**Resource Management Pattern**
```python
# Clean startup/shutdown lifecycle
async def __aenter__(self):
    await self.initialize()
    return self

async def __aexit__(self, exc_type, exc_val, exc_tb):
    await self.cleanup()  # Always cleanup server processes

# Usage:
async with MCPClient("filesystem", config) as client:
    tools = await client.list_tools()
    result = await client.execute_tool("read_file", {"path": "/etc/hosts"})
    # Auto-cleanup happens here
```

##### **🔄 Multi-Client Orchestration**

**How Multiple Clients Work Together**
```python
# In ChatSession - the orchestrator
class ChatSession:
    def __init__(self, clients: List[MCPClient]):
        self.clients = clients  # e.g., [filesystem_client, datetime_client, web_client]
        self.tool_routing = {}  # tool_name -> which_client
    
    async def initialize(self):
        # 1. Initialize each client separately
        for client in self.clients:
            await client.initialize()
            
            # 2. Build unified tool map
            for tool in await client.list_tools():
                self.tool_routing[tool.name] = client
        
        # Result: {"read_file": filesystem_client, 
        #          "get_time": datetime_client,
        #          "web_search": web_client}
    
    async def execute_any_tool(self, tool_name, args):
        # Route to correct client automatically
        client = self.tool_routing[tool_name]
        return await client.execute_tool(tool_name, args)
```

##### **🛠️ Real Usage Example**

```python
# Complete flow from user query to tool execution
async def handle_user_request():
    # Setup: Multiple servers for different capabilities (auto-transport detection)
    config = {
        "filesystem": {"command": "npx", "args": ["@modelcontextprotocol/server-filesystem"]},
        "datetime": {"url": "http://127.0.0.1:8001/mcp", "timeout": 30},  # HTTP server
        "markdown": {"command": "python", "args": ["src/servers/markdown.py"]},  # STDIO server
    }
    
    # Create clients (one per server) - auto-detects transport type
    fs_client = MCPClient("filesystem", config["filesystem"])  # STDIO
    dt_client = MCPClient("datetime", config["datetime"])      # HTTP
    md_client = MCPClient("markdown", config["markdown"])      # STDIO
    
    # Initialize connections (transport handled automatically)
    await fs_client.initialize()  # ← Starts filesystem server via STDIO
    await dt_client.initialize()  # ← Connects to HTTP datetime server
    await md_client.initialize()  # ← Starts markdown server via STDIO
    
    # Now we have access to all tools from all servers
    fs_tools = await fs_client.list_tools()    # ["read_file", "write_file", "list_dir"]
    dt_tools = await dt_client.list_tools()    # ["get_time", "format_date"]
    md_tools = await md_client.list_tools()    # ["read_markdown", "write_markdown"]
    
    # Execute tools through appropriate clients (same API regardless of transport)
    files = await fs_client.execute_tool("list_dir", {"path": "/home"})
    time = await dt_client.execute_tool("get_time", {"timezone": "UTC"})
    content = await md_client.execute_tool("read_markdown", {"path": "README.md"})
```

**Why This Architecture?**
- ✅ **Isolation**: Each server runs independently, can't interfere with others
- ✅ **Composability**: Add/remove servers without affecting others  
- ✅ **Security**: Each server has its own permissions and scope
- ✅ **Reliability**: If one server fails, others keep working
- ✅ **Scalability**: Easy to add new capabilities by adding new servers

#### 3. **Chat Session Orchestrator** (`src/mcp/session.py`)
The intelligent session manager that coordinates between users, LLMs, and tools:
- **Core Function**: Orchestrates multi-turn conversations with dynamic tool integration
- **Key Features**:
  - Automatic tool call detection and extraction from LLM responses
  - Parallel tool execution across multiple MCP servers
  - Response streaming with real-time workflow visualization
  - Error handling and retry mechanisms
  - Message history management

#### 4. **LLM Integration Layer** (`src/model/`)
Pluggable LLM client architecture supporting multiple providers:
- **Current Support**: OpenAI (GPT-4o, GPT-4o-mini)
- **Planned**: Anthropic Claude, Google Gemini
- **Features**: Standardized interface for seamless provider switching

#### 5. **MCP Server Layer** (`src/mcp/servers/`)

**🖥️ Core Responsibility: Lightweight Programs Exposing Specific Capabilities**

MCP servers are standalone programs that run in separate processes and expose tools, resources, and prompts through the standardized MCP protocol.

##### **🏗️ Basic Server Communication Flow**

```mermaid
sequenceDiagram
    participant Host as Host Application
    participant Client as MCP Client
    participant Server as MCP Server Process
    
    Host->>Client: Need to call tool "read_file"
    Client->>Server: Start process (python server.py)
    Server-->>Client: MCP handshake + capabilities
    Client->>Server: tools/list request
    Server-->>Client: Available tools ["read_file", "write_file"]
    Client->>Server: tools/call "read_file" {path: "data.txt"}
    Server-->>Client: File contents
    Client-->>Host: Tool result
```

##### **💡 Core Server Concepts**

**Tools - Functions LLMs Can Execute**
```python
@mcp.tool()
def calculate(operation: str, a: float, b: float) -> dict:
    """Perform basic math operations"""
    if operation == "add":
        return {"result": a + b}
    elif operation == "multiply":
        return {"result": a * b}
    else:
        return {"error": "Unknown operation"}
```

**Resources - Data Access Points**
```python
# Resources are like "files" the LLM can read
# Examples: configuration files, API data, database records
@mcp.resource("config://settings")
async def get_settings():
    return {"theme": "dark", "language": "en"}
```

**Prompts - Reusable Templates** 
```python
# Pre-written prompts for common tasks
@mcp.prompt("summarize_code")
def code_summary_prompt(code: str):
    return f"Please summarize this code and identify any issues:\n\n{code}"
```

##### **🔧 Creating a Simple Server**

**Step 1: Basic Server Structure**
```python
# my_server.py
from mcp.server.fastmcp import FastMCP

# Initialize server
server = FastMCP("my-calculator")

@server.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together"""
    return a + b

@server.tool() 
def get_time() -> str:
    """Get current time"""
    from datetime import datetime
    return datetime.now().isoformat()

# Run server
if __name__ == "__main__":
    server.run()  # Starts STDIO communication
```

**Step 2: Add to Configuration**
```json
{
    "mcpServers": {
        "calculator": {
            "command": "python",
            "args": ["my_server.py"]
        }
    }
}
```

**Step 3: Server Automatically Available**
```python
# Host application automatically discovers tools:
# - add_numbers(a, b) 
# - get_time()
# 
# LLM can now use these tools in conversations
```

##### **🛠️ Server Development Patterns**

**Error Handling Best Practice**
```python
@server.tool()
def safe_file_read(filename: str) -> dict:
    """Read file with error handling"""
    try:
        # Validate input
        if not filename.endswith('.txt'):
            return {"error": "Only .txt files allowed"}
        
        # Perform operation  
        with open(filename, 'r') as f:
            content = f.read()
        
        return {"success": True, "content": content}
        
    except FileNotFoundError:
        return {"error": f"File {filename} not found"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}
```

**Async Operations**
```python
@server.tool()
async def fetch_data(url: str) -> dict:
    """Fetch data from external API"""
    import httpx
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=10.0)
            return {"data": response.json(), "status": response.status_code}
        except Exception as e:
            return {"error": str(e)}
```

##### **🔄 Server Lifecycle in Host**

**How Servers Integrate**
```python
# Host application startup process:
async def initialize_mcp_ecosystem():
    servers = {}
    
    # 1. Read server configuration
    config = load_servers_config()
    
    # 2. Start each server process
    for name, server_config in config.items():
        # Start: python my_server.py  
        process = start_server_process(server_config)
        
        # 3. MCP handshake via STDIO
        client = MCPClient(name, process)
        await client.initialize()
        
        # 4. Discover available tools
        tools = await client.list_tools()
        
        servers[name] = {
            "client": client,
            "tools": tools  # ["add_numbers", "get_time"]
        }
    
    return servers  # Ready for LLM to use!
```

**Tool Routing**
```python
# When LLM wants to call a tool:
async def execute_tool(tool_name: str, arguments: dict):
    # 1. Find which server has this tool
    for server_name, server_info in servers.items():
        if tool_name in [t.name for t in server_info["tools"]]:
            # 2. Route to correct server
            client = server_info["client"]
            result = await client.execute_tool(tool_name, arguments)
            return result
    
    raise ValueError(f"Tool {tool_name} not found")
```

**Why This Architecture?**
- ✅ **Isolation**: Each server runs separately, failures don't cascade
- ✅ **Language Flexibility**: Write servers in Python, Node.js, Go, etc.
- ✅ **Simple Protocol**: Just JSON over STDIO - easy to implement
- ✅ **Auto-Discovery**: Tools automatically appear in host application
- ✅ **Composability**: Mix different servers for different capabilities

### Protocol Implementation

This implementation follows the [MCP specification](mcp-docs/sections/specification.md) and includes:

#### **Capability Negotiation**
As defined in the [MCP architecture](mcp-docs/sections/architecture.md), clients and servers explicitly declare supported features:

```python
# Example from our implementation
async def initialize(self) -> None:
    """Initialize server connection with capability negotiation"""
    server_params = StdioServerParameters(
        command=command,
        args=self.config["args"],
        env=environment_vars
    )
    session = await ClientSession(read, write)
    await session.initialize()  # Performs capability negotiation
```

#### **Tool Discovery and Execution**
Following the [MCP tools specification](mcp-docs/sections/tools.md):

```python
async def list_tools(self) -> List[MCPTool]:
    """Discover available tools with metadata"""
    tools_response = await self.session.list_tools()
    return [MCPTool(tool.name, tool.description, tool.inputSchema) 
            for tool in tools_response]

async def execute_tool(self, tool_name: str, arguments: dict) -> Any:
    """Execute tool with retry mechanism and error handling"""
    return await self.session.call_tool(tool_name, arguments)
```

## 🚀 Key Features

### 1. **Multi-Server Integration**
- Simultaneous connections to multiple MCP servers
- Automatic tool discovery and consolidation
- Load balancing and failover capabilities
- Mix STDIO and HTTP servers in the same application

### 2. **Intelligent Tool Orchestration**  
- Automatic tool call detection from natural language
- JSON-based tool invocation with schema validation
- Parallel tool execution for improved performance
- Comprehensive error handling and retries

### 3. **Real-Time Workflow Visualization**
- Live streaming of LLM responses
- Interactive workflow step display
- Tool execution progress tracking
- Detailed error reporting and debugging

### 4. **Secure Multi-Tenant Architecture**
- Isolated client connections per server
- Configurable security policies
- Resource cleanup and connection management
- Session state isolation

### 5. **Extensible Provider Support**
- Pluggable LLM provider architecture
- Runtime provider switching
- Standardized configuration management

## 📁 Project Structure

```
ubs-mcp-poc/
├── chatbot.py                 # Main Streamlit application (Host)
├── servers_config.json        # MCP server configuration (STDIO & HTTP)
├── run_http_servers.py        # Helper script to run HTTP servers
├── example_http_usage.py      # HTTP transport examples
├── environment.yml           # Conda environment setup
├── src/
│   ├── mcp/                  # MCP implementation
│   │   ├── client.py         # MCP client implementation
│   │   ├── session.py        # Chat session orchestrator
│   │   ├── tool.py          # Tool abstraction layer
│   │   └── servers/         # Custom MCP servers
│   │       ├── markdown_processor.py
│   │       └── datetime_processor.py
│   ├── model/               # LLM integration layer
│   │   ├── factory.py       # Provider factory
│   │   ├── base.py         # Base LLM interface
│   │   └── oai.py          # OpenAI implementation
│   ├── config.py           # Configuration management
│   ├── workflow.py         # Workflow visualization
│   └── consts.py           # Project constants
└── mcp-docs/               # MCP protocol documentation
    └── sections/           # Detailed specification docs
```

## ⚙️ Configuration

### MCP Server Configuration (`servers_config.json`)

The configuration supports both STDIO and HTTP transports with automatic detection:

```json
{
    "mcpServers": {
        "markdown_processor": {
            "url": "http://127.0.0.1:8000/mcp",
            "timeout": 30
        },
        "datetime_processor": {
            "url": "http://127.0.0.1:8001/mcp",
            "timeout": 30
        },
        "filesystem": {
            "command": "npx",
            "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/directory"]
        }
    }
}
```

**Transport Auto-Detection:**
- **HTTP**: Configuration contains `"url"` → connects to running HTTP server
- **STDIO**: Configuration contains `"command"` → starts server as subprocess

**HTTP Configuration Options:**
- `url`: Server endpoint (required)
- `headers`: Optional HTTP headers for authentication
- `timeout`: Request timeout in seconds (default: 30)

### Environment Setup
```bash
# Create conda environment
conda env create -f environment.yml
conda activate ubs-mcp-poc

# Install additional dependencies
pip install mcp[cli] openai-agents python-dotenv streamlit
```

## 🎯 Usage

### Option 1: HTTP Transport (Recommended for Production)

1. **Start HTTP MCP Servers:**
   ```bash
   # Start all servers as persistent HTTP services
   python run_http_servers.py
   ```

2. **Run the Chatbot:**
   ```bash
   streamlit run chatbot.py
   ```

### Option 2: STDIO Transport (Good for Development)

1. **Update Configuration:** Change `servers_config.json` to use STDIO:
   ```json
   {
       "mcpServers": {
           "markdown_processor": {
               "command": "python",
               "args": ["src/mcp/servers/markdown_processor.py"]
           }
       }
   }
   ```

2. **Run the Chatbot:**
   ```bash
   streamlit run chatbot.py
   ```

### Testing HTTP Transport
```bash
# Test HTTP transport functionality
python example_http_usage.py
```

### Example Interactions

1. **File Operations**
   - "What files are in the root directory?"
   - "Create a new markdown file with project documentation"

2. **Data Processing** 
   - "Process this markdown content and extract headers"
   - "Convert this text to a structured format"

3. **Time-Based Operations**
   - "What's the current time in UTC?"
   - "Schedule a reminder for next week"

### Tool Call Flow

The system automatically detects when tools are needed and executes them transparently:

1. **User Input**: Natural language query
2. **LLM Processing**: Determines if tools are needed
3. **Tool Discovery**: Finds relevant tools across all connected servers
4. **Execution**: Calls appropriate tools with extracted parameters
5. **Response Generation**: Processes results into natural language

## 🔧 How MCP Works in This Implementation

### Connection Lifecycle

Following the [MCP lifecycle specification](mcp-docs/sections/lifecycle.md):

1. **Initialization Phase**
   ```python
   # Host creates clients for each configured server
   for name, config in server_configs.items():
       client = MCPClient(name, config)
       await client.initialize()  # Establishes connection
   ```

2. **Capability Exchange**
   ```python
   # Server declares its capabilities
   await session.initialize()  # Negotiates supported features
   tools = await session.list_tools()  # Discovers available tools
   ```

3. **Active Session**
   ```python
   # Normal operation with requests/responses
   result = await session.call_tool(tool_name, arguments)
   ```

4. **Cleanup**
   ```python
   # Proper resource cleanup
   await client.cleanup()
   ```

### Message Flow

Our implementation follows the standard [MCP message patterns](mcp-docs/sections/messages.md):

```mermaid
sequenceDiagram
    participant User
    participant ChatSession
    participant MCPClient
    participant MCPServer

    User->>ChatSession: "What files are available?"
    ChatSession->>MCPClient: list_tools()
    MCPClient->>MCPServer: tools/list
    MCPServer-->>MCPClient: Available tools
    MCPClient-->>ChatSession: Tool metadata
    ChatSession->>MCPClient: call_tool("list_files", {})
    MCPClient->>MCPServer: tools/call
    MCPServer-->>MCPClient: File listing
    MCPClient-->>ChatSession: Results
    ChatSession-->>User: "Here are the available files..."
```

## 🔐 Security Considerations

Following [MCP security best practices](mcp-docs/sections/security-best-practices.md):

### 1. **Server Isolation**
- Each MCP client maintains separate connections
- No cross-server data leakage
- Individual security policies per server

### 2. **Tool Authorization**
- Human-in-the-loop tool execution approval
- Clear visual indicators for tool invocations
- Configurable security policies

### 3. **Data Protection**
- Local data processing when possible
- Minimal context sharing with servers
- Secure credential management
