# Snowflake MCP Server Browser Authentication Implementation Plan

## Project Goal

Create an AI agent capable of connecting to Snowflake databases using browser-based authentication while maintaining all existing functionality of the MCP Snowflake server. This enables seamless integration with enterprise SSO systems without storing credentials directly.

## Key Challenges

1. **Browser Authentication Flow**: Snowflake's `browserauthentication` method opens a browser window for user login, which requires handling an interactive process.

2. **Token Management**: Securely storing and reusing authentication tokens to avoid repeated browser prompts.

3. **Session Expiration**: Gracefully handling session timeouts and token refreshes.

4. **Error Handling**: Managing authentication failures and retry mechanisms.

5. **Backward Compatibility**: Ensuring existing functionality continues to work while adding browser authentication.

## Implementation Strategy

### 1. Authentication Manager Module

**File**: `auth_manager.py`

**Purpose**:
- Manage the authentication lifecycle
- Store and retrieve tokens securely
- Handle token expiration and renewal

**Key Components**:
- `TokenManager`: Securely stores and retrieves tokens with encryption
- `BrowserAuthManager`: Coordinates browser authentication process

**Advantages**:
- Separation of concerns from database logic
- Reusable token management system
- Secure storage with encryption

### 2. Enhancements to SnowflakeDB Class

**File**: `server.py` (modify existing class)

**Changes**:
- Add support for detecting browser authentication request
- Integrate with auth manager when needed
- Handle token retrieval and refresh
- Maintain backward compatibility for existing authentication methods

**Advantages**:
- Minimal changes to core functionality
- Transparent to higher-level code
- Falls back to standard auth if browser auth fails

### 3. ~~Response Format Standardization~~ (Removed)

We've decided not to modify the response format since:
- The existing format is optimized for large datasets from Snowflake
- Maintaining backward compatibility with existing clients is critical
- The current format already provides sufficient information for the agent

## Implementation Details

### 1. Token Management

The token management system will:

- **Encrypt tokens** using machine-specific keys
- **Store tokens** in a secure local file
- **Set appropriate permissions** on token storage files
- **Validate tokens** before use
- **Track expiration** to proactively refresh

### 2. Browser Authentication Process

The authentication flow will:

1. Check for cached token first
2. Use token if available and valid
3. Fall back to browser authentication if needed
4. Launch browser with SSO login page
5. Capture and store token after successful login
6. Handle authentication errors gracefully

### 3. Integration with Existing Code

We'll integrate with the existing code by:

- Modifying the `_init_database` method to support browser auth
- Adding token validation before query execution
- Setting up token refresh on session timeout
- Maintaining all tool handlers without major changes

## Implementation Phases

### Phase 1: Auth Manager Implementation

- Create the `auth_manager.py` module
- Implement token storage and retrieval
- Add encryption for secure token handling

### Phase 2: SnowflakeDB Integration

- Modify `SnowflakeDB.__init__` to accept auth parameters
- Update `_init_database` to use browser auth when specified
- Add token validation and renewal logic

### Phase 3: Testing & Refinement

- Test with actual Snowflake instances
- Verify token reuse works correctly
- Ensure backward compatibility

## Benefits to MCP SQL Agent Project

1. **Enterprise Integration**: Connect to enterprise Snowflake instances using SSO without storing credentials

2. **Improved Security**: No passwords stored in configuration files or environment variables

3. **User Experience**: One-time authentication prompt, then seamless operation

4. **Cross-database Joins**: Maintain ability to join data across datasources with enhanced authentication

5. **Consistent API**: All existing agent functionality continues to work with the enhanced authentication

## Implementation Considerations

### Security

- Tokens are encrypted at rest
- Encryption keys are derived from machine-specific information
- File permissions restrict access to token cache

### User Experience

- Browser authentication happens only once per session
- Clear error messages for authentication issues
- Graceful token refresh when sessions expire

### Configuration

- Add new parameters:
  - `--authenticator browserauthentication`
  - `--token_cache_path` (optional path to token cache)

### Backward Compatibility

- Standard username/password authentication will continue to work
- All existing tools and resources remain functional
- No change to response formats by default

## Next Steps After Implementation

1. **Documentation**: Update README with browser authentication instructions

2. **Testing**: Validate with different SSO providers and Snowflake instances

3. **Integration**: Update MCP agent to leverage the enhanced server capabilities

4. **Extensions**: Consider adding support for other authentication methods

This implementation plan provides a structured approach to adding browser authentication to the Snowflake MCP server while maintaining backward compatibility and following security best practices.