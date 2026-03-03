import asyncio
import json
import logging
from typing import Any

from aidial_client import AsyncDial
from aidial_client.types.chat import Message as ClientChatMessage
from aidial_sdk.chat_completion import (
    Choice,
    Message,
    Request,
    Response,
    Role,
    ToolCall,
)

from task.tools.base import BaseTool
from task.tools.models import ToolCallParams
from task.utils.constants import TOOL_CALL_HISTORY_KEY
from task.utils.history import unpack_messages
from task.utils.stage import StageProcessor

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class GeneralPurposeAgent:
    def __init__(
        self,
        endpoint: str,
        system_prompt: str,
        tools: list[BaseTool],
    ):
        self.endpoint = endpoint
        self.system_prompt = system_prompt
        self.tools = tools
        self.tools_dict = {tool.name: tool for tool in tools}
        self.state = {TOOL_CALL_HISTORY_KEY: []}

    async def handle_request(
        self, deployment_name: str, choice: Choice, request: Request, response: Response
    ) -> Message:
        # TODO:
        # 1. Create AsyncDial, don't forget to provide endpoint as base_url and api_key. Api_key you can take from `request` as well as api_version
        #    JFI: while request you will get Per-request API key (not `dial_api_key` configured in Core config). Read
        #    more about it -> https://docs.dialx.ai/platform/core/per-request-keys
        client = AsyncDial(
            base_url=self.endpoint,
            api_key=request.api_key,
            api_version=request.api_version,
        )
        # 2. Create `chunks` with created AsyncDial client (chat -> completions -> create). Provide it with:
        #    - messages: get messages from `request` and unpack them with `_prepare_messages` method
        #    - tools: provide list with tool schemas
        #    - deployment_name
        #    - make it stream
        chunks = await client.chat.completions.create(
            deployment_name=deployment_name,
            messages=self._prepare_messages(request.messages),
            tools=[tool.schema for tool in self.tools],
            stream=True,
        )
        # 3. Create:
        #   - `tool_call_index_map` (it is empty dict), here we will collect tool calls by their indexes.
        #      Take a look how tool call streaming output is looks like, it is important! -> https://platform.openai.com/docs/guides/function-calling#streaming
        #   - `content`, here we will collect the content from streaming
        tool_call_index_map = {}
        content = ""
        # 4. Make async loop through `chunks` and then we need to collect content, tool calls and attachments:
        #   - If chunk has `choices` then:
        #       - Get 1st choice `delta`
        #       - if delta is present:
        #           - if delta content is present then append this content to `choice` (it will be shown in DIAL Chat
        #             choice), concat delta content to `content` variable
        #           - if delta has tool_calls then:
        #               - iterate through tool_calls:
        #                   - if tool call has `id` (first chunk of tool call) then add to `tool_call_index_map` new
        #                     tool_call_delta, key will be index and value tool call delta itself
        #                   - otherwise: get by tool call delta `index` from the `tool_call_index_map` the tool call and
        #                     then check if provided tool_call_delta contains `function`, if yes then you need to get from
        #                     `function` `arguments` (if not present set them as empty string to not attach haphazardly None)
        #                     as `argument_chunk` and add it to the extracted from map tool_call function arguments
        async for chunk in chunks:
            if chunk.choices:
                if delta := chunk.choices[0].delta:
                    if delta.content:
                        choice.append_content(delta.content)
                        content += delta.content
                    if delta.tool_calls:
                        for tool_call_delta in delta.tool_calls:
                            if tool_call_delta.id:
                                tool_call_index_map[tool_call_delta.index] = (
                                    tool_call_delta
                                )
                            else:
                                tool_call = tool_call_index_map[tool_call_delta.index]
                                if tool_call_delta.function:
                                    argument_chunk = (
                                        tool_call_delta.function.arguments or ""
                                    )
                                    tool_call.function.arguments += argument_chunk
        # 5. Create `assistant_message`, with role, content and tool_calls. `tool_calls` should be a list with ToolCall
        #    objects generated from `tool_call_index_map` dict values. to create ToolCall use `validate` method (it
        #    will show you the notification that it is deprecated but we need to use it because DIAL SDK is built on top of pydentic.v1)
        assistant_message = Message(role=Role.ASSISTANT, content=content)
        assistant_message.tool_calls = [
            ToolCall.validate(tool_call) for tool_call in tool_call_index_map.values()
        ]
        log.debug("assistant_message: " + json.dumps(assistant_message.dict()))

        # 6. Now we at the point where we need to understand if its 'final result' from orchestration model or not:
        #    check if `assistant_message` contains `tool_calls`, if yes then we need:
        #       - create `tasks` list. Iterate through `tool_calls` and call `_process_tool_call` method (do not use
        #         `await` since we will run tool calls execution asynchronously), also you need to provide `conversation_id`
        #         you can get it in `request` headers, its name is `x-conversation-id`
        #       - now `gather` tasks with `asyncio` (here you need to await)
        #       - to the `state` to `TOOL_CALL_HISTORY_KEY` append `assistant_message` as dict and exclude none from this dict
        #       - extend the `state` `TOOL_CALL_HISTORY_KEY` with tool_messages that we executed above
        #       - finally make recursive call
        if assistant_message.tool_calls:
            conversation_id = request.headers.get("x-conversation-id")

            if not conversation_id:
                raise ValueError("x-conversation-id header is required")

            tasks = [
                self._process_tool_call(
                    tool_call, choice, request.api_key, conversation_id
                )
                for tool_call in assistant_message.tool_calls
            ]

            tool_messages = await asyncio.gather(*tasks)

            self.state[TOOL_CALL_HISTORY_KEY].append(
                assistant_message.dict(exclude_none=True)
            )
            self.state[TOOL_CALL_HISTORY_KEY].extend(tool_messages)

            return await self.handle_request(
                deployment_name=deployment_name,
                choice=choice,
                request=request,
                response=response,
            )

        # 7. We don't have any tool calls and reasy to finish user request. Set choice with `state` and return `assistant_message`
        choice.set_state(self.state)

        return assistant_message

    def _prepare_messages(self, messages: list[Message]) -> list[ClientChatMessage]:
        unpacked_messages = [
            {"role": Role.SYSTEM, "content": self.system_prompt},
            *unpack_messages(messages, self.state[TOOL_CALL_HISTORY_KEY]),
        ]

        for message in unpacked_messages:
            print(json.dumps(message, indent=2))

        return unpacked_messages

    async def _process_tool_call(
        self, tool_call: ToolCall, choice: Choice, api_key: str, conversation_id: str
    ) -> dict[str, Any]:
        # TODO:
        # 1. Get tool name from tool_call function name
        tool_name = tool_call.function.name
        log.debug(f"tool_call: {tool_call.dict()}")
        # 2. Open Stage with StageProcessor (it will be shown in DIAL Chat and Stage serves in our case for
        #    tool call results representation)
        stage = StageProcessor.open_stage(choice, tool_name)
        # 3. Get tool from `_tools_dict` by tool name
        tool = self.tools_dict.get(tool_name)
        # 4. If tool show_in_stage is true then:
        #   - append content to stage "## Request arguments: \n"
        #   - append content to stage f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r"
        #     it will print arguments as markdown json
        #   - append content to stage "## Response: \n"
        if not tool:
            raise ValueError(f"Tool {tool_name} not found")

        if tool.show_in_stage:
            stage.append_content("## Request arguments: \n")
            stage.append_content(
                f"```json\n\r{json.dumps(json.loads(tool_call.function.arguments), indent=2)}\n\r```\n\r"
            )
            stage.append_content("## Response: \n")
        # 5. Execute tool
        tool_message = await tool.execute(
            ToolCallParams(
                tool_call=tool_call,
                stage=stage,
                choice=choice,
                api_key=api_key,
                conversation_id=conversation_id,
            )
        )
        # 6. Close stage with StageProcessor
        StageProcessor.close_stage_safely(stage)
        # 7. Return tool message as dict and don't forget to exclude none
        return tool_message.dict(exclude_none=True)
