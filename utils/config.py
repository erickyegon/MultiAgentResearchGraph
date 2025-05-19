"""
Configuration utilities for the Knowledge Graph Builder application.

This module provides functions for managing API keys and other configuration settings.
"""

import os
import logging
import json
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

# Constants
CONFIG_FILE = Path("config/app_config.json")
DEFAULT_CONFIG = {
    "api_keys": {
        "serp_api_key": "",
        "euri_api_key": ""
    },
    "app_settings": {
        "max_results": 10,
        "default_graph_layout": "spring",
        "show_categories": True,
        "show_relationships": True
    }
}


def ensure_config_dir() -> None:
    """Ensure the config directory exists."""
    os.makedirs(CONFIG_FILE.parent, exist_ok=True)


def load_config() -> Dict[str, Any]:
    """
    Load configuration from the config file.

    Returns:
        Dict[str, Any]: The configuration dictionary
    """
    ensure_config_dir()

    if not CONFIG_FILE.exists():
        # Create default config file if it doesn't exist
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.error(f"Error loading config file: {e}")
        return DEFAULT_CONFIG.copy()


def save_config(config: Dict[str, Any]) -> bool:
    """
    Save configuration to the config file.

    Args:
        config (Dict[str, Any]): The configuration to save

    Returns:
        bool: True if successful, False otherwise
    """
    ensure_config_dir()

    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except IOError as e:
        logger.error(f"Error saving config file: {e}")
        return False


def set_api_keys(serp_api_key: str, euri_api_key: str) -> bool:
    """
    Set API keys in both environment variables and config file.

    Args:
        serp_api_key (str): The SERP API key
        euri_api_key (str): The EURI API key

    Returns:
        bool: True if successful, False otherwise
    """
    # Set environment variables
    os.environ["SERP_API_KEY"] = serp_api_key
    os.environ["EURI_API_KEY"] = euri_api_key

    # Update config file
    config = load_config()
    config["api_keys"]["serp_api_key"] = serp_api_key
    config["api_keys"]["euri_api_key"] = euri_api_key

    return save_config(config)


def get_api_keys() -> Tuple[Optional[str], Optional[str]]:
    """
    Get API keys from environment variables or config file.

    Returns:
        Tuple[Optional[str], Optional[str]]: (SERP API key, EURI API key)
    """
    # Try environment variables first
    serp_api_key = os.environ.get("SERP_API_KEY")
    euri_api_key = os.environ.get("EURI_API_KEY")

    # If not found, try config file
    if not serp_api_key or not euri_api_key:
        config = load_config()
        api_keys = config.get("api_keys", {})

        if not serp_api_key:
            serp_api_key = api_keys.get("serp_api_key")

        if not euri_api_key:
            euri_api_key = api_keys.get("euri_api_key")

    return serp_api_key, euri_api_key


def check_api_keys() -> bool:
    """
    Check if API keys are set.

    Returns:
        bool: True if both API keys are set, False otherwise
    """
    serp_api_key, euri_api_key = get_api_keys()
    return bool(serp_api_key and euri_api_key)


def get_app_settings() -> Dict[str, Any]:
    """
    Get application settings from config file.

    Returns:
        Dict[str, Any]: Application settings
    """
    config = load_config()
    return config.get("app_settings", DEFAULT_CONFIG["app_settings"])


def update_app_settings(settings: Dict[str, Any]) -> bool:
    """
    Update application settings in config file.

    Args:
        settings (Dict[str, Any]): The settings to update

    Returns:
        bool: True if successful, False otherwise
    """
    config = load_config()
    config["app_settings"].update(settings)
    return save_config(config)
