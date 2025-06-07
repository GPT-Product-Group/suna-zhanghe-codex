import os
import json
import base64
import mimetypes
from typing import Optional, Union, List

import httpx

from agentpress.tool import ToolResult, openapi_schema, xml_schema
from sandbox.tool_base import SandboxToolsBase
from agentpress.thread_manager import ThreadManager
from utils.config import config

mimetypes.add_type("audio/mpeg", ".mp3")

class DifyChatflowTool(SandboxToolsBase):
    """Tool for invoking a Dify ChatFlow workflow."""

    def __init__(self, project_id: str, thread_id: str, thread_manager: ThreadManager):
        super().__init__(project_id, thread_manager)
        self.thread_id = thread_id
        self.api_key = config.get("DIFY_API_KEY")
        self.chatflow_url = config.get("DIFY_CHATFLOW_URL")
        if not self.api_key or not self.chatflow_url:
            raise ValueError("DIFY_API_KEY and DIFY_CHATFLOW_URL must be configured")

    def _encode_file(self, path: str) -> Optional[dict]:
        """Read and base64 encode a file from the sandbox."""
        try:
            cleaned = self.clean_path(path)
            full_path = f"{self.workspace_path}/{cleaned}"
            file_bytes = self.sandbox.fs.download_file(full_path)
            mime, _ = mimetypes.guess_type(full_path)
            if not mime:
                mime = "application/octet-stream"
            b64 = base64.b64encode(file_bytes).decode("utf-8")
            return {"file_name": os.path.basename(cleaned), "base64": b64, "mime_type": mime}
        except Exception:
            return None

    @openapi_schema({
        "type": "function",
        "function": {
            "name": "run_chatflow",
            "description": "Run the configured Dify chatflow with the given text input and optional attachments (images or audio).",
            "parameters": {
                "type": "object",
                "properties": {
                    "input_text": {
                        "type": "string",
                        "description": "Main text input to pass to the chatflow"
                    },
                    "attachments": {
                        "anyOf": [
                            {"type": "string"},
                            {"items": {"type": "string"}, "type": "array"}
                        ],
                        "description": "Optional file paths (relative to /workspace) for images or audio to include"
                    }
                },
                "required": ["input_text"]
            }
        }
    })
    @xml_schema(
        tag_name="run-chatflow",
        mappings=[
            {"param_name": "input_text", "node_type": "content", "path": "."},
            {"param_name": "attachments", "node_type": "attribute", "path": ".", "required": False}
        ],
        example='''
        <run-chatflow attachments="images/picture.png">Describe this image</run-chatflow>
        '''
    )
    async def run_chatflow(self, input_text: str, attachments: Optional[Union[str, List[str]]] = None) -> ToolResult:
        """Invoke the Dify ChatFlow workflow and store the result."""
        try:
            await self._ensure_sandbox()
            if attachments and isinstance(attachments, str):
                attachments = [attachments]

            files_payload = []
            if attachments:
                for p in attachments:
                    data = self._encode_file(p)
                    if data:
                        files_payload.append(data)
                    else:
                        return self.fail_response(f"Could not read attachment: {p}")

            payload = {"inputs": {"text": input_text}}
            if files_payload:
                payload["files"] = files_payload

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

            async with httpx.AsyncClient() as client:
                resp = await client.post(self.chatflow_url, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                result = resp.json()

            await self.thread_manager.add_message(self.thread_id, "dify_input", payload, is_llm_message=False)
            await self.thread_manager.add_message(self.thread_id, "dify_output", result, is_llm_message=False)

            text = result.get("answer") or result.get("message") or json.dumps(result)
            return self.success_response(text)
        except httpx.HTTPError as e:
            return self.fail_response(f"HTTP error: {str(e)}")
        except Exception as e:
            return self.fail_response(f"Error running chatflow: {str(e)}")
