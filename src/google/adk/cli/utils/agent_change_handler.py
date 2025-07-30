# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""File system event handler for agent changes to trigger hot reload for agents."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from watchdog.events import FileSystemEventHandler

from .agent_loader import AgentLoader
from .shared_value import SharedValue

logger = logging.getLogger("google_adk." + __name__)


class AgentChangeEventHandler(FileSystemEventHandler):

  def __init__(
      self,
      agent_loader: AgentLoader,
      runners_to_clean: set[str],
      current_app_name_ref: SharedValue[str],
      memory_service_ref: SharedValue,  # Add reference to memory service
      agents_dir: str,  # Add agents directory path
  ):
    self.agent_loader = agent_loader
    self.runners_to_clean = runners_to_clean
    self.current_app_name_ref = current_app_name_ref
    self.memory_service_ref = memory_service_ref
    self.agents_dir = agents_dir

  def _update_memory_service_from_config(self, changed_file_path: str):
    """Update memory service based on config.json in the changed directory."""
    try:
      # Get the directory containing the changed file
      changed_dir = Path(changed_file_path).parent
      
      # Only proceed if the directory name starts with 'memory_tool'
      if not changed_dir.name.startswith('memory_tool'):
        logger.debug(f"Directory {changed_dir.name} does not start with 'memory_tool', skipping memory service update")
        return
      
      # Look for config.json in the same directory as the changed file
      config_path = changed_dir / "rag_tool_config.json"
      
      if config_path.exists():
        with open(config_path, 'r') as f:
          config = json.load(f)
        
        # Extract RESOURCE_ID from config
        resource_config = config.get("config")
        if not resource_config:
          logger.warning(f"Config not found in rag_tool_config.json at {config_path}")
          return
        
        resource_id = resource_config.get("resourceId")
        
        if resource_id:
          # Import here to avoid circular imports
          from google.adk.memory.vertex_ai_rag_memory_service import VertexAiRagMemoryService
          import os
          from . import envs
          
          # Load environment variables for the agent
          agent_name = changed_dir.name
          envs.load_dotenv_for_agent(agent_name, self.agents_dir)
          
          # Create new memory service with updated RAG corpus
          new_memory_service = VertexAiRagMemoryService(
              rag_corpus=resource_id
          )
          
          # Update the shared memory service reference
          self.memory_service_ref.value = new_memory_service
          
          logger.info(f"Updated memory service with RAG corpus: {resource_id} from directory: {changed_dir.name}")
        else:
          logger.warning(f"RESOURCE_ID not found in config.json at {config_path}")
      else:
        logger.debug(f"No config.json found in {changed_dir}")
        
    except Exception as e:
      logger.error(f"Error updating memory service from config: {e}")

  def on_modified(self, event):
    if not (event.src_path.endswith(".py") or event.src_path.endswith(".yaml")):
      return
    logger.info("Change detected in agents directory: %s", event.src_path)
    
    # Update memory service if config.json exists
    self._update_memory_service_from_config(event.src_path)
    
    # Original functionality
    self.agent_loader.remove_agent_from_cache(self.current_app_name_ref.value)
    self.runners_to_clean.add(self.current_app_name_ref.value)
