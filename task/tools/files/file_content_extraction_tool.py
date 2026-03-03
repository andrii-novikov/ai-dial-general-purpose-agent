import json
from typing import Any

from aidial_sdk.chat_completion import Message

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.dial_file_conent_extractor import DialFileContentExtractor


class FileContentExtractionTool(BaseTool):
    """
    Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM.
    PAGINATION: Files >10,000 chars are paginated. Response format: `**Page #X. Total pages: Y**` appears at end if paginated.
    USAGE: Start with page=1 (by default)
    """

    def __init__(self, endpoint: str):
        self.endpoint = endpoint

    @property
    def show_in_stage(self) -> bool:
        # TODO: set as False since we will have custom variant of representation in Stage
        return False

    @property
    def name(self) -> str:
        # TODO: provide self-descriptive name
        return "file_content_extraction_tool"

    @property
    def description(self) -> str:
        # TODO: provide tool description that will help LLM to understand when to use this tools and cover 'tricky'
        #  moments (not more 1024 chars)
        return """
        "Extracts text content from files. Supported: PDF (text only), TXT, CSV (as markdown table), HTML/HTM.
        PAGINATION: Files >10,000 chars are paginated. Response format: `**Page #X. Total pages: Y**` appears at end if paginated.
        USAGE: Start with page=1 (by default)"
        """

    @property
    def parameters(self) -> dict[str, Any]:
        # TODO: provide tool parameters JSON Schema:
        #  - file_url is string, required
        #  - page is integer, by default 1, description: "For large documents pagination is enabled. Each page consists of 10000 characters."
        return {
            "type": "object",
            "properties": {
                "file_url": {
                    "type": "string",
                    "description": "URL of the file to extract content from.",
                },
                "page": {
                    "type": "integer",
                    "default": 1,
                    "description": "For large documents pagination is enabled. Each page consists of 10000 characters.",
                },
            },
            "required": ["file_url"],
        }

    async def _execute(self, tool_call_params: ToolCallParams) -> str | Message:
        # TODO:
        # 1. Load arguments with `json`
        arguments = json.loads(tool_call_params.tool_call.function.arguments)
        # 2. Get `file_url` from arguments
        file_url = arguments.get("file_url")
        # 3. Get `page` from arguments (if none, set as 1 by default)
        page = arguments.get("page", 1)
        # 4. Get stage from `tool_call_params`
        stage = tool_call_params.stage
        # 5. Append content to stage: "## Request arguments: \n"
        stage.append_content(f"## Request arguments: \n")
        # 6. Append content to stage: `f"**File URL**: {file_url}\n\r"`
        stage.append_content(f"**File URL**: {file_url}\n\r")
        # 7. If `page` more than 1 then append content to stage: `f"**Page**: {page}\n\r"`
        if page > 1:
            stage.append_content(f"**Page**: {page}\n\r")
        # 8. Append content to stage: "## Response: \n"
        stage.append_content(f"## Response: \n")
        # 9. Implement `task.utils.dial_file_conent_extractor`, create DialFileContentExtractor and call `extract_text`
        #    method as `content`
        content = DialFileContentExtractor(
            self.endpoint, tool_call_params.api_key
        ).extract_text(file_url)
        # 10. If no `content` present then set it as "Error: File content not found."
        if not content:
            content = "Error: File content not found."
        # 11. If `content` len is more than 10_000 then we need to enable pagination:
        #       - create variable `page_size` as 10_000
        #       - calculate total pages, formula: (`content len` + `page_size` - 1) // `page_size`
        #       - if `page` is less then 1 (potential hallucination from LLM) then set it as 1
        #       - otherwise check if page > total pages (potential hallucination), it yes then set `content` as
        #         `f"Error: Page {page} does not exist. Total pages: {total_pages}"`
        #       - prepare `start_index`: `(page - 1) * page_size`
        #       - prepare `end_index`: `start_index + page_size`
        #       - get page content from `content` that will start with `start_index` and end with `end_index`
        #       - set `content` as `f"{page_content}\n\n**Page #{page}. Total pages: {total_pages}**"` (It will show to
        #         LLM that it is not full content and it is pageable)
        if len(content) > 10_000:
            page_size = 10_000
            total_pages = (len(content) + page_size - 1) // page_size
            if page < 1:
                page = 1
            if page > total_pages:
                content = (
                    f"Error: Page {page} does not exist. Total pages: {total_pages}"
                )
            else:
                start_index = (page - 1) * page_size
                end_index = start_index + page_size
                content = content[start_index:end_index]
                content = f"{content}\n\n**Page #{page}. Total pages: {total_pages}**"
        # 12. Append content to stage: `f"```text\n\r{content}\n\r```\n\r"` (Will be shown in stage as markdown text)
        stage.append_content(f"```text\n\r{content}\n\r```\n\r")
        # 13. Return `content`
        return content
