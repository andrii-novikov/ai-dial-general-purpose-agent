from dataclasses import dataclass

from aidial_sdk.chat_completion import Choice, Stage, ToolCall

# from aidial_client.types.chat.legacy.chat_completion import ToolCall


@dataclass
class ToolCallParams:
    tool_call: ToolCall
    stage: Stage
    choice: Choice
    api_key: str
    conversation_id: str
