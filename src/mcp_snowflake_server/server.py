import importlib.metadata
import json
import logging
import os
import time
import uuid
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import mcp.server.stdio
import mcp.types as types
import yaml
import pandas as pd
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from pydantic import AnyUrl, BaseModel
from snowflake.snowpark import Session

from .write_detector import SQLWriteDetector
from .response_formatter import (
    StandardResponse, 
    format_dataframe_result,
    format_table_list,
    format_table_schema,
    format_error_response,
    format_write_operation_result
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("mcp_snowflake_server")


def data_to_yaml(data: Any) -> str:
    return yaml.dump(data, indent=2, sort_keys=False)


class SnowflakeDB:
    AUTH_EXPIRATION_TIME = 1800

    def __init__(self, connection_config: dict, token_cache_path: Optional[str] = None):
        self.connection_config = connection_config.copy()
        self.session = None
        self.snowflake_connection = None
        self.insights: list[str] = []
        self.auth_time = 0
        
        # Set up browser authentication if enabled
        self.use_browser_auth = (
            self.connection_config.get("authenticator", "").lower() == "browserauthentication"
        )
        
        # Initialize browser auth manager if needed
        self.auth_manager = None
        if self.use_browser_auth:
            from .auth_manager import BrowserAuthManager
            self.auth_manager = BrowserAuthManager(token_cache_path)
            logger.info("Browser authentication enabled")

    def _init_database(self):
        """Initialize connection to the Snowflake database"""
        try:
            # Apply browser authentication if configured
            if self.use_browser_auth and self.auth_manager:
                # Prepare connection parameters with browser auth or token
                auth_params = self.auth_manager.prepare_connection_parameters(self.connection_config)
                logger.info(f"Using authentication method: {auth_params.get('authenticator', 'default')}")
                
                # Create connection
                self.session = Session.builder.configs(auth_params).create()
                
                # Get actual Snowflake connection from Snowpark session
                self.snowflake_connection = self.session._conn._conn
                
                # Save token if browser auth was successful
                if auth_params.get("authenticator") == "browserauthentication":
                    self.auth_manager.handle_successful_auth(
                        auth_params, self.snowflake_connection
                    )
            else:
                # Standard authentication
                self.session = Session.builder.configs(self.connection_config).create()
                self.snowflake_connection = self.session._conn._conn
                logger.info("Using standard authentication")
            
            # Set database context
            for component in ["database", "schema", "warehouse"]:
                if component in self.connection_config:
                    self.session.sql(f"USE {component.upper()} {self.connection_config[component].upper()}")
            
            # Update auth time
            self.auth_time = time.time()
            logger.info("Database connection initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake database: {e}")
            # If browser auth is enabled and we had a cached token that failed,
            # clear the token and try again
            if (
                self.use_browser_auth 
                and self.auth_manager 
                and "authenticator" in self.connection_config
                and self.connection_config["authenticator"] == "oauth"
            ):
                logger.info("Token authentication failed, clearing token and retrying with browser authentication")
                self.auth_manager.clear_token(self.connection_config)
                
                # Reset authenticator to browserauthentication and try again
                self.connection_config["authenticator"] = "browserauthentication"
                self._init_database()  # Recursive call with browserauthentication
                return
                
            raise ValueError(f"Failed to connect to Snowflake database: {e}")

    def execute_query(self, query: str) -> Tuple[pd.DataFrame, str]:
        """
        Execute a SQL query and return results as a pandas DataFrame and data_id.
        
        Args:
            query: SQL query to execute
            
        Returns:
            Tuple of (DataFrame, data_id)
        """
        # Check if session needs to be initialized or refreshed
        if not self.session or time.time() - self.auth_time > self.AUTH_EXPIRATION_TIME:
            logger.info("Session expired or not initialized, reconnecting...")
            self._init_database()

        logger.debug(f"Executing query: {query}")
        try:
            # Execute query
            result_df = self.session.sql(query).to_pandas()
            data_id = str(uuid.uuid4())
            
            # Return DataFrame and data_id
            return result_df, data_id

        except Exception as e:
            logger.error(f'Database error executing "{query}": {e}')
            raise

    def execute_write_query(self, query: str) -> Tuple[int, str]:
        """
        Execute a write query (INSERT, UPDATE, DELETE) and return affected rows.
        
        Args:
            query: SQL write query to execute
            
        Returns:
            Tuple of (affected_rows, data_id)
        """
        # Check if session needs to be initialized or refreshed
        if not self.session or time.time() - self.auth_time > self.AUTH_EXPIRATION_TIME:
            self._init_database()

        logger.debug(f"Executing write query: {query}")
        try:
            # Execute query
            result = self.session.sql(query)
            
            # Get number of affected rows (stored in rowcount attribute)
            affected_rows = 0
            if hasattr(result, '_df') and hasattr(result._df, '_rows_affected'):
                affected_rows = result._df._rows_affected
            
            # Generate data ID
            data_id = str(uuid.uuid4())
            
            logger.info(f"Write operation affected {affected_rows} rows")
            return affected_rows, data_id

        except Exception as e:
            logger.error(f'Database error executing write query "{query}": {e}')
            raise

    def add_insight(self, insight: str) -> None:
        """Add a new insight to the collection"""
        self.insights.append(insight)
        logger.info(f"Added new insight: {insight[:50]}...")

    def get_memo(self) -> str:
        """Generate a formatted memo from collected insights"""
        if not self.insights:
            return "No data insights have been discovered yet."

        memo = "📊 Data Intelligence Memo 📊\n\n"
        memo += "Key Insights Discovered:\n\n"
        memo += "\n".join(f"- {insight}" for insight in self.insights)

        if len(self.insights) > 1:
            memo += f"\n\nSummary:\nAnalysis has revealed {len(self.insights)} key data insights that suggest opportunities for strategic optimization and growth."

        return memo


def handle_tool_errors(func: Callable) -> Callable:
    """Decorator to standardize tool error handling"""

    @wraps(func)
    async def wrapper(*args, **kwargs) -> list[types.TextContent]:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            return [types.TextContent(type="text", text=f"Error: {str(e)}")]

    return wrapper


class Tool(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[str, dict[str, Any] | None], list[types.TextContent | types.ImageContent | types.EmbeddedResource]]
    tags: list[str] = []


# Tool handlers
async def handle_list_tables(arguments, db, *_):
    """Handle list_tables tool request with standardized response format."""
    schema = db.connection_config['schema'].upper()
    database = db.connection_config['database']
    
    # Construct and execute query
    query = f"""
        SELECT table_catalog, table_schema, table_name, comment 
        FROM {database}.information_schema.tables 
        WHERE table_schema = '{schema}'
    """
    
    try:
        # Execute query and get DataFrame
        df, data_id = db.execute_query(query)
        
        # Format response using our standard formatter
        response = format_table_list(
            tables=df.to_dict(orient="records"),
            query=query,
            schema=schema
        )
        
        # Convert to YAML and JSON for the response
        yaml_output = response.to_yaml()
        json_output = response.to_json()
        
        # Return both formats
        return [
            types.TextContent(type="text", text=yaml_output),
            types.EmbeddedResource(
                type="resource",
                resource=types.TextResourceContents(
                    uri=f"data://{data_id}", 
                    text=json_output, 
                    mimeType="application/json"
                ),
            ),
        ]
    except Exception as e:
        # Format error response
        error_response = format_error_response(
            error=e,
            query=query,
            help_text="Error occurred while listing tables. Check your database connection and permissions."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]


async def handle_describe_table(arguments, db, *_):
    """Handle describe_table tool request with standardized response format."""
    if not arguments or "table_name" not in arguments:
        raise ValueError("Missing table_name argument")

    # Parse table name parts
    split_identifier = arguments["table_name"].split(".")
    table_name = split_identifier[-1].upper()
    schema_name = (split_identifier[-2] if len(split_identifier) > 1 else db.connection_config["schema"]).upper()
    database_name = (split_identifier[-3] if len(split_identifier) > 2 else db.connection_config["database"]).upper()

    # Construct query
    query = f"""
        SELECT column_name, column_default, is_nullable, data_type, comment 
        FROM {database_name}.information_schema.columns 
        WHERE table_schema = '{schema_name}' AND table_name = '{table_name}'
    """
    
    try:
        # Execute query
        df, data_id = db.execute_query(query)
        
        # Format response
        response = format_table_schema(
            columns=df.to_dict(orient="records"),
            table_name=f"{schema_name}.{table_name}",
            query=query
        )
        
        # Convert to YAML and JSON
        yaml_output = response.to_yaml()
        json_output = response.to_json()
        
        # Return both formats
        return [
            types.TextContent(type="text", text=yaml_output),
            types.EmbeddedResource(
                type="resource",
                resource=types.TextResourceContents(
                    uri=f"data://{data_id}", 
                    text=json_output, 
                    mimeType="application/json"
                ),
            ),
        ]
    except Exception as e:
        # Format error response
        error_response = format_error_response(
            error=e,
            query=query,
            help_text=f"Error describing table {table_name}. Check that the table exists and you have permission to access it."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]


async def handle_read_query(arguments, db, write_detector, *_):
    """Handle read_query tool request with standardized response format."""
    query = arguments["query"]
    
    # Check for write operations
    if write_detector.analyze_query(query)["contains_write"]:
        error_response = format_error_response(
            error="Calls to read_query should not contain write operations",
            query=query,
            help_text="This tool is for read-only queries. Use write_query for modifications."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]
    
    try:
        # Execute query
        df, data_id = db.execute_query(query)
        
        # Format response
        response = format_dataframe_result(
            df=df,
            query=query,
            help_text="Query executed successfully."
        )
        
        # Convert to YAML and JSON
        yaml_output = response.to_yaml()
        json_output = response.to_json()
        
        # Return both formats
        return [
            types.TextContent(type="text", text=yaml_output),
            types.EmbeddedResource(
                type="resource",
                resource=types.TextResourceContents(
                    uri=f"data://{data_id}", 
                    text=json_output, 
                    mimeType="application/json"
                ),
            ),
        ]
    except Exception as e:
        # Format error response
        error_response = format_error_response(
            error=e,
            query=query,
            help_text="Error executing query. Check your syntax and table names."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]


async def handle_append_insight(arguments, db, _, __, server):
    """Handle append_insight tool request."""
    if not arguments or "insight" not in arguments:
        error_response = format_error_response(
            error="Missing insight argument",
            help_text="You must provide an 'insight' parameter with the insight text."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]

    try:
        # Add insight to the database
        db.add_insight(arguments["insight"])
        
        # Notify clients that the resource was updated
        await server.request_context.session.send_resource_updated(AnyUrl("memo://insights"))
        
        # Create success response
        response = StandardResponse(
            status="success",
            message="Insight added to memo successfully",
            help_text="The insight has been added to the data intelligence memo.",
            metadata={"insight_count": len(db.insights)}
        )
        
        return [types.TextContent(type="text", text=response.to_yaml())]
    except Exception as e:
        # Format error response
        error_response = format_error_response(
            error=e,
            help_text="Error adding insight to memo."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]


async def handle_write_query(arguments, db, _, allow_write, __):
    """Handle write_query tool request with standardized response format."""
    query = arguments["query"]
    
    # Check if write operations are allowed
    if not allow_write:
        error_response = format_error_response(
            error="Write operations are not allowed for this data connection",
            query=query,
            help_text="This server is configured for read-only access. Enable with --allow_write flag."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]
    
    # Check if the query is a SELECT (should use read_query instead)
    if query.strip().upper().startswith("SELECT"):
        error_response = format_error_response(
            error="SELECT queries are not allowed for write_query",
            query=query,
            help_text="Use the read_query tool for SELECT queries."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]

    try:
        # Detect operation type
        operation_type = "write"
        if query.strip().upper().startswith("INSERT"):
            operation_type = "insert"
        elif query.strip().upper().startswith("UPDATE"):
            operation_type = "update"
        elif query.strip().upper().startswith("DELETE"):
            operation_type = "delete"
        
        # Execute query and get affected rows
        affected_rows, data_id = db.execute_write_query(query)
        
        # Format response
        response = format_write_operation_result(
            affected_rows=affected_rows,
            query=query,
            operation_type=operation_type
        )
        
        # Return response
        return [types.TextContent(type="text", text=response.to_yaml())]
    except Exception as e:
        # Format error response
        error_response = format_error_response(
            error=e,
            query=query,
            help_text="Error executing write query. Check your syntax and permissions."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]


async def handle_create_table(arguments, db, _, allow_write, __):
    """Handle create_table tool request with standardized response format."""
    query = arguments["query"]
    
    # Check if write operations are allowed
    if not allow_write:
        error_response = format_error_response(
            error="Write operations are not allowed for this data connection",
            query=query,
            help_text="This server is configured for read-only access. Enable with --allow_write flag."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]
    
    # Check if the query is a CREATE TABLE statement
    if not query.strip().upper().startswith("CREATE TABLE"):
        error_response = format_error_response(
            error="Only CREATE TABLE statements are allowed",
            query=query,
            help_text="This tool only supports CREATE TABLE statements."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]

    try:
        # Extract table name from query
        # This is a simplistic approach - in a real implementation, you might use a SQL parser
        query_parts = query.strip().split()
        table_index = query_parts.index("TABLE") if "TABLE" in query_parts else -1
        table_name = query_parts[table_index + 1] if table_index >= 0 and table_index + 1 < len(query_parts) else "unknown_table"
        
        # Remove schema prefix if present
        if "." in table_name:
            table_name = table_name.split(".")[-1]
        
        # Execute query
        _, data_id = db.execute_query(query)
        
        # Format response
        response = StandardResponse(
            status="success",
            message=f"Table {table_name} created successfully",
            query=query,
            table_name=table_name,
            help_text="Table creation completed successfully."
        )
        
        # Return response
        return [types.TextContent(type="text", text=response.to_yaml())]
    except Exception as e:
        # Format error response
        error_response = format_error_response(
            error=e,
            query=query,
            help_text="Error creating table. Check your syntax and permissions."
        )
        return [types.TextContent(type="text", text=error_response.to_yaml())]


async def prefetch_tables(db: SnowflakeDB, credentials: dict) -> dict:
    """Prefetch table and column information"""
    try:
        logger.info("Prefetching table descriptions")
        table_results, data_id = db.execute_query(
            f"""SELECT table_name, comment 
                FROM {credentials['database']}.information_schema.tables 
                WHERE table_schema = '{credentials['schema'].upper()}'"""
        )

        column_results, data_id = db.execute_query(
            f"""SELECT table_name, column_name, data_type, comment 
                FROM {credentials['database']}.information_schema.columns 
                WHERE table_schema = '{credentials['schema'].upper()}'"""
        )

        tables_brief = {}
        for row in table_results:
            tables_brief[row["TABLE_NAME"]] = {**row, "COLUMNS": {}}

        for row in column_results:
            row_without_table_name = row.copy()
            del row_without_table_name["TABLE_NAME"]
            tables_brief[row["TABLE_NAME"]]["COLUMNS"][row["COLUMN_NAME"]] = row_without_table_name

        return tables_brief

    except Exception as e:
        logger.error(f"Error prefetching table descriptions: {e}")
        return f"Error prefetching table descriptions: {e}"


async def main(
    allow_write: bool = False,
    connection_args: dict = None,
    log_dir: str = None,
    prefetch: bool = False,
    log_level: str = "INFO",
    exclude_tools: list[str] = [],
    token_cache_path: str = None,
):
    # Setup logging
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        logger.handlers.append(logging.FileHandler(os.path.join(log_dir, "mcp_snowflake_server.log")))
    if log_level:
        logger.setLevel(log_level)

    logger.info("Starting Snowflake MCP Server")
    logger.info("Allow write operations: %s", allow_write)
    logger.info("Prefetch table descriptions: %s", prefetch)
    logger.info("Excluded tools: %s", exclude_tools)
    
    # Check if browser authentication is enabled
    browser_auth_enabled = connection_args.get("authenticator", "").lower() == "browserauthentication"
    if browser_auth_enabled:
        logger.info("Browser authentication enabled")
        if token_cache_path:
            logger.info(f"Using token cache path: {token_cache_path}")
        else:
            logger.info("Using default token cache path")
    
    # Initialize the database connection
    db = SnowflakeDB(connection_args, token_cache_path=token_cache_path)
    server = Server("snowflake-manager")
    write_detector = SQLWriteDetector()

    tables_info = (await prefetch_tables(db, connection_args)) if prefetch else {}
    tables_brief = data_to_yaml(tables_info) if prefetch else ""

    all_tools = [
        Tool(
            name="list_tables",
            description="List all tables in the Snowflake database",
            input_schema={
                "type": "object",
                "properties": {},
            },
            handler=handle_list_tables,
            tags=["description"],
        ),
        Tool(
            name="describe_table",
            description="Get the schema information for a specific table",
            input_schema={
                "type": "object",
                "properties": {"table_name": {"type": "string", "description": "Name of the table to describe"}},
                "required": ["table_name"],
            },
            handler=handle_describe_table,
            tags=["description"],
        ),
        Tool(
            name="read_query",
            description="Execute a SELECT query.",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "SELECT SQL query to execute"}},
                "required": ["query"],
            },
            handler=handle_read_query,
        ),
        Tool(
            name="append_insight",
            description="Add a data insight to the memo",
            input_schema={
                "type": "object",
                "properties": {"insight": {"type": "string", "description": "Data insight discovered from analysis"}},
                "required": ["insight"],
            },
            handler=handle_append_insight,
            tags=["resource_based"],
        ),
        Tool(
            name="write_query",
            description="Execute an INSERT, UPDATE, or DELETE query on the Snowflake database",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "SQL query to execute"}},
                "required": ["query"],
            },
            handler=handle_write_query,
            tags=["write"],
        ),
        Tool(
            name="create_table",
            description="Create a new table in the Snowflake database",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "CREATE TABLE SQL statement"}},
                "required": ["query"],
            },
            handler=handle_create_table,
            tags=["write"],
        ),
    ]

    exclude_tags = []
    if not allow_write:
        exclude_tags.append("write")
    if prefetch:
        exclude_tags.append("description")
    allowed_tools = [
        tool for tool in all_tools if tool.name not in exclude_tools and not any(tag in exclude_tags for tag in tool.tags)
    ]

    logger.info("Allowed tools: %s", [tool.name for tool in allowed_tools])

    # Register handlers
    @server.list_resources()
    async def handle_list_resources() -> list[types.Resource]:
        resources = [
            types.Resource(
                uri=AnyUrl("memo://insights"),
                name="Data Insights Memo",
                description="A living document of discovered data insights",
                mimeType="text/plain",
            )
        ]
        table_brief_resources = [
            types.Resource(
                uri=AnyUrl(f"context://table/{table_name}"),
                name=f"{table_name} table",
                description=f"Description of the {table_name} table",
                mimeType="text/plain",
            )
            for table_name in tables_info.keys()
        ]
        resources += table_brief_resources
        return resources

    @server.read_resource()
    async def handle_read_resource(uri: AnyUrl) -> str:
        if str(uri) == "memo://insights":
            return db.get_memo()
        elif str(uri).startswith("context://table"):
            table_name = str(uri).split("/")[-1]
            if table_name in tables_info:
                return data_to_yaml(tables_info[table_name])
            else:
                raise ValueError(f"Unknown table: {table_name}")
        else:
            raise ValueError(f"Unknown resource: {uri}")

    @server.list_prompts()
    async def handle_list_prompts() -> list[types.Prompt]:
        return []

    @server.get_prompt()
    async def handle_get_prompt(name: str, arguments: dict[str, str] | None) -> types.GetPromptResult:
        raise ValueError(f"Unknown prompt: {name}")

    @server.call_tool()
    @handle_tool_errors
    async def handle_call_tool(
        name: str, arguments: dict[str, Any] | None
    ) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
        if name in exclude_tools:
            return [types.TextContent(type="text", text=f"Tool {name} is excluded from this data connection")]

        handler = next((tool.handler for tool in allowed_tools if tool.name == name), None)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")

        return await handler(arguments, db, write_detector, allow_write, server)

    @server.list_tools()
    async def handle_list_tools() -> list[types.Tool]:
        logger.info("Listing tools")
        logger.error(f"Allowed tools: {allowed_tools}")
        tools = [
            types.Tool(
                name=tool.name,
                description=tool.description,
                inputSchema=tool.input_schema,
            )
            for tool in allowed_tools
        ]
        return tools

    # Start server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        logger.info("Server running with stdio transport")
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="snowflake",
                server_version=importlib.metadata.version("mcp_snowflake_server"),
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
