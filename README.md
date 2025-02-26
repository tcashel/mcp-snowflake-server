# Snowflake MCP Server

[![smithery badge](https://smithery.ai/badge/mcp_snowflake_server)](https://smithery.ai/server/mcp_snowflake_server) [![PyPI - Version](https://img.shields.io/pypi/dm/mcp-snowflake-server?color&logo=pypi&logoColor=white&label=PyPI%20downloads)](https://pypi.org/project/mcp-snowflake-server/)


## Overview
A Model Context Protocol (MCP) server implementation that provides database interaction with Snowflake. This server enables running SQL queries with tools and interacting with a memo of data insights presented as a resource.

## Authentication Methods

The server supports multiple authentication methods:

1. **Username/Password Authentication** - Standard username and password login
2. **Browser Authentication** - Uses Snowflake's browser-based authentication for SSO systems
3. **OAuth Token** - Uses cached tokens from previous browser authentications

When using browser authentication:
- A browser window will open for you to log in with your SSO credentials
- After successful login, the token is securely cached for future use
- Subsequent connections will use the cached token without requiring browser login
- Tokens are automatically refreshed when they expire

## Components

### Resources
The server exposes a single dynamic resource:
- `memo://insights`: A continuously updated data insights memo that aggregates discovered insights during analysis
  - Auto-updates as new insights are discovered via the append-insight tool

### Tools
The server offers six core tools:

#### Query Tools
- `read_query`
   - Execute SELECT queries to read data from the database
   - Input:
     - `query` (string): The SELECT SQL query to execute
   - Returns: Query results as array of objects

- `write_query` (with `--allow-write` flag)
   - Execute INSERT, UPDATE, or DELETE queries
   - Input:
     - `query` (string): The SQL modification query
   - Returns: `{ affected_rows: number }`

- `create_table` (with `--allow-write` flag)
   - Create new tables in the database
   - Input:
     - `query` (string): CREATE TABLE SQL statement
   - Returns: Confirmation of table creation

#### Schema Tools
- `list_tables`
   - Get a list of all tables in the database
   - No input required
   - Returns: Array of table names

- `describe-table`
   - View column information for a specific table
   - Input:
     - `table_name` (string): Name of table to describe (can be fully qualified)
   - Returns: Array of column definitions with names and types

#### Analysis Tools
- `append_insight`
   - Add new data insights to the memo resource
   - Input:
     - `insight` (string): data insight discovered from analysis
   - Returns: Confirmation of insight addition
   - Triggers update of memo://insights resource


## Usage with Claude Desktop

### Installing via Smithery

To install Snowflake Server for Claude Desktop automatically via [Smithery](https://smithery.ai/server/mcp_snowflake_server):

```bash
npx -y @smithery/cli install mcp_snowflake_server --client claude
```

### Using Standard Authentication (via UVX)

```python
# Add the server to your claude_desktop_config.json with standard authentication
"mcpServers": {
  "snowflake_std": {
      "command": "uvx",
      "args": [
          "mcp_snowflake_server",
          "--account",
          "the_account",
          "--warehouse",
          "the_warehouse",
          "--user",
          "the_user",
          "--password",
          "their_password",
          "--role",
          "the_role"
          "--database",
          "the_database",
          "--schema",
          "the_schema",
          # Optionally: "--allow_write" (but not recommended)
          # Optionally: "--log_dir", "/absolute/path/to/logs"
          # Optionally: "--log_level", "DEBUG"/"INFO"/"WARNING"/"ERROR"/"CRITICAL"
          # Optionally: "--exclude_tools", "{tool name}", ["{other tool name}"]
      ]
  }
}
```

### Using Browser Authentication (via UVX)

```python
# Add the server to your claude_desktop_config.json with browser authentication
"mcpServers": {
  "snowflake_browser": {
      "command": "uvx",
      "args": [
          "mcp_snowflake_server",
          "--account",
          "the_account",
          "--warehouse",
          "the_warehouse",
          "--user",
          "the_user",
          "--role",
          "the_role",
          "--database",
          "the_database",
          "--schema",
          "the_schema",
          "--authenticator",
          "browserauthentication",
          # Optionally: "--token_cache_path", "/path/to/token/cache",
          # Optionally: "--allow_write" (but not recommended)
          # Optionally: "--log_dir", "/absolute/path/to/logs"
          # Optionally: "--log_level", "DEBUG"/"INFO"/"WARNING"/"ERROR"/"CRITICAL"
          # Optionally: "--exclude_tools", "{tool name}", ["{other tool name}"]
      ]
  }
}
```

### Installing Locally

```python
# Add the server to your claude_desktop_config.json
"mcpServers": {
  "snowflake_local": {
      "command": "uv",
      "args": [
          "--directory",
          "/absolute/path/to/mcp_snowflake_server",
          "run",
          "mcp_snowflake_server",
          "--account",
          "the_account",
          "--warehouse",
          "the_warehouse",
          "--user",
          "the_user",
          # Choose one of these authentication methods:
          "--password",
          "their_password",
          # OR
          "--authenticator",
          "browserauthentication",
          "--role",
          "the_role"
          "--database",
          "the_database",
          "--schema",
          "the_schema",
          # Optionally: "--token_cache_path", "/path/to/token/cache"
          # Optionally: "--allow_write" (but not recommended)
          # Optionally: "--log_dir", "/absolute/path/to/logs"
          # Optionally: "--log_level", "DEBUG"/"INFO"/"WARNING"/"ERROR"/"CRITICAL"
          # Optionally: "--exclude_tools", "{tool name}", ["{other tool name}"]
      ]
  }
}
```

## Browser Authentication Details

When using browser authentication:

1. The first time you connect, a browser window will open automatically
2. Log in with your SSO credentials in the browser
3. The authentication token will be securely stored (encrypted) on your machine
4. Future connections will use the stored token without opening a browser
5. If the token expires, a new browser window will open for reauthentication

You can specify a custom location for the token cache:
```
--token_cache_path "/path/to/your/token/cache"
```

The default location is `~/.snowflake_tokens`.
