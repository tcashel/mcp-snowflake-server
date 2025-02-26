"""
Authentication Manager for Snowflake MCP Server

This module manages authentication for the Snowflake MCP server,
including browser-based authentication, token caching, and token refresh.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Configure logging
logger = logging.getLogger("mcp_snowflake_auth")


class TokenManager:
    """Manages Snowflake authentication tokens, including caching and encryption."""

    # Default location for token cache file
    DEFAULT_TOKEN_CACHE_PATH = "~/.snowflake_tokens"
    
    # Salt for encryption key derivation
    DEFAULT_SALT = b"snowflake-mcp-server-salt"
    
    def __init__(self, token_cache_path: Optional[str] = None, machine_key: Optional[str] = None):
        """
        Initialize the token manager.
        
        Args:
            token_cache_path: Path to the token cache file
            machine_key: Optional machine-specific key for encryption
        """
        # Set token cache path
        self.token_cache_path = os.path.expanduser(token_cache_path or self.DEFAULT_TOKEN_CACHE_PATH)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.token_cache_path), exist_ok=True)
        
        # Set up encryption
        self._setup_encryption(machine_key)
        
        # Load tokens from cache
        self.tokens = self._load_tokens()
        
        logger.debug(f"Token manager initialized with cache path: {self.token_cache_path}")

    def _setup_encryption(self, machine_key: Optional[str] = None):
        """Set up encryption for token storage."""
        # Use machine_key or generate one based on machine info
        if not machine_key:
            # Use machine ID or hostname as a unique identifier
            try:
                with open("/etc/machine-id", "r") as f:
                    machine_id = f.read().strip()
            except FileNotFoundError:
                import socket
                machine_id = socket.gethostname()
                
            machine_key = f"snowflake-mcp-{machine_id}"
        
        # Generate encryption key from machine key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.DEFAULT_SALT,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(machine_key.encode()))
        self.cipher = Fernet(key)
    
    def _load_tokens(self) -> Dict[str, Dict[str, Any]]:
        """Load tokens from the cache file."""
        if not os.path.exists(self.token_cache_path):
            return {}
        
        try:
            with open(self.token_cache_path, "rb") as f:
                encrypted_data = f.read()
                
            # Decrypt the data
            decrypted_data = self.cipher.decrypt(encrypted_data).decode("utf-8")
            return json.loads(decrypted_data)
        except Exception as e:
            logger.warning(f"Error loading tokens from cache: {e}")
            return {}
    
    def _save_tokens(self):
        """Save tokens to the cache file with encryption."""
        try:
            # Convert tokens to JSON
            json_data = json.dumps(self.tokens)
            
            # Encrypt the data
            encrypted_data = self.cipher.encrypt(json_data.encode("utf-8"))
            
            # Save to file with secure permissions
            with open(self.token_cache_path, "wb") as f:
                f.write(encrypted_data)
                
            # Set file permissions to be readable only by the owner
            os.chmod(self.token_cache_path, 0o600)
            
            logger.debug(f"Tokens saved to {self.token_cache_path}")
        except Exception as e:
            logger.error(f"Error saving tokens to cache: {e}")
    
    def get_token(self, account: str, user: str) -> Optional[Dict[str, Any]]:
        """
        Get a token for the given account and user.
        
        Args:
            account: Snowflake account identifier
            user: Snowflake username
        
        Returns:
            Dictionary containing token information if found, None otherwise
        """
        key = f"{account}:{user}"
        token_info = self.tokens.get(key)
        
        if token_info:
            # Check if token is expired
            if token_info.get("expiry", 0) < time.time():
                logger.info(f"Token for {key} has expired")
                return None
            
            logger.debug(f"Found valid token for {key}")
            return token_info
        
        return None
    
    def save_token(self, account: str, user: str, token: Dict[str, Any], expiration_seconds: int = 3600):
        """
        Save a token for the given account and user.
        
        Args:
            account: Snowflake account identifier
            user: Snowflake username
            token: Token information to save
            expiration_seconds: Token expiration time in seconds
        """
        key = f"{account}:{user}"
        
        # Add expiry timestamp
        token_info = token.copy()
        token_info["expiry"] = time.time() + expiration_seconds
        token_info["saved_at"] = datetime.now().isoformat()
        
        # Save token
        self.tokens[key] = token_info
        self._save_tokens()
        
        logger.info(f"Saved token for {key} with expiration in {expiration_seconds}s")
    
    def delete_token(self, account: str, user: str):
        """
        Delete a token for the given account and user.
        
        Args:
            account: Snowflake account identifier
            user: Snowflake username
        """
        key = f"{account}:{user}"
        if key in self.tokens:
            del self.tokens[key]
            self._save_tokens()
            logger.info(f"Deleted token for {key}")


class BrowserAuthManager:
    """Manages Snowflake browser authentication processes."""
    
    def __init__(self, token_cache_path: Optional[str] = None):
        """
        Initialize the browser authentication manager.
        
        Args:
            token_cache_path: Path to the token cache file
        """
        self.token_manager = TokenManager(token_cache_path)
        logger.info("Browser authentication manager initialized")
    
    def prepare_connection_parameters(self, connection_config: Dict[str, str]) -> Dict[str, str]:
        """
        Prepare connection parameters for Snowflake with browser authentication.
        
        Args:
            connection_config: Original connection configuration
        
        Returns:
            Updated connection configuration with browser authentication
        """
        # Make a copy to avoid modifying the original
        params = connection_config.copy()
        
        # Check if authenticator is already set to browser authentication
        if params.get("authenticator", "").lower() != "browserauthentication":
            # Set authenticator to browserauthentication
            params["authenticator"] = "browserauthentication"
            logger.info("Setting authenticator to browserauthentication")
        
        # Ensure required parameters are available
        required_params = ["account", "user", "warehouse", "role"]
        missing_params = [param for param in required_params if param not in params]
        
        if missing_params:
            raise ValueError(f"Missing required parameters for browser authentication: {missing_params}")
        
        # Get token if available
        token_info = self.token_manager.get_token(params["account"], params["user"])
        
        if token_info:
            logger.info(f"Using cached token for {params['user']}@{params['account']}")
            # Add token to parameters
            params["token"] = token_info.get("token")
            params["authenticator"] = "oauth"
        else:
            logger.info(f"No valid token found, will use browser authentication")
        
        return params
    
    def handle_successful_auth(self, connection_config: Dict[str, str], connection_object: Any):
        """
        Handle successful authentication by saving the token.
        
        Args:
            connection_config: Connection configuration
            connection_object: Snowflake connection object
        """
        try:
            # Extract account and user
            account = connection_config.get("account")
            user = connection_config.get("user")
            
            if not account or not user:
                logger.warning("Missing account or user in connection config, cannot save token")
                return
            
            # Extract token from connection
            if hasattr(connection_object, "get_oauth_token"):
                token = connection_object.get_oauth_token()
                
                if token:
                    # Save token
                    token_info = {"token": token}
                    self.token_manager.save_token(account, user, token_info)
                    logger.info(f"Successfully saved token for {user}@{account}")
                else:
                    logger.warning("No token available from connection")
            else:
                logger.warning("Connection object does not support oauth token extraction")
        except Exception as e:
            logger.error(f"Error saving token after successful authentication: {e}")
    
    def clear_token(self, connection_config: Dict[str, str]):
        """
        Clear the token for the given connection configuration.
        
        Args:
            connection_config: Connection configuration
        """
        account = connection_config.get("account")
        user = connection_config.get("user")
        
        if account and user:
            self.token_manager.delete_token(account, user)
            logger.info(f"Cleared token for {user}@{account}")