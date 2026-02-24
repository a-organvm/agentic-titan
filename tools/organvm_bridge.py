"""
OrganVM Bridge - Auto-loads the local organvm-mcp-server into Titan.
"""

from __future__ import annotations

import logging
import shutil
import sys
import os
from pathlib import Path
from typing import Any

from titan.core.config import get_config
from tools.base import get_registry
from tools.mcp_client import MCPClient

logger = logging.getLogger("titan.tools.organvm")


async def load_organvm_tools() -> None:
    """
    Connect to the local organvm-mcp-server and register its tools.
    
    This gives Titan agents "hands" to reach into the registry and graph.
    """
    # Try to find organvm-mcp in PATH or current venv
    executable = shutil.which("organvm-mcp")
    
    if not executable:
        # Check current venv bin
        venv_bin = Path(sys.prefix) / "bin" / "organvm-mcp"
        if venv_bin.exists():
            executable = str(venv_bin)
            
    if not executable:
        logger.warning("organvm-mcp executable not found. Skipping system awareness tools.")
        return

    logger.info(f"Connecting to organvm-mcp-server at {executable}...")
    
    client = MCPClient(command=executable)
    try:
        await client.connect()
        tools = await client.get_tools()
        
        registry = get_registry()
        for tool in tools:
            registry.register(tool)
            
        logger.info(f"Successfully registered {len(tools)} tools from OrganVM MCP.")
        
    except Exception as e:
        logger.error(f"Failed to load OrganVM tools: {e}")
