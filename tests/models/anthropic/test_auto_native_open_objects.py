"""A bare structured `output_type` with a `dict` field keeps Tool Output on a request that thinks.

On a request that thinks, a bare structured `output_type` uses Native Output, so the forced output tool call
doesn't turn thinking off. Native Output is always strict, and Anthropic's strict schema transformation sets
`additionalProperties` to `false` on every object, so the model can only return `{}` for a `dict` field. Seen on
`claude-haiku-5-5`, which thinks by default: every `dict` field of the output came back empty.
"""

from __future__ import annotations as _annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel

from pydantic_ai import Agent
from pydantic_ai.output import NativeOutput

from ..._inline_snapshot import snapshot
from ...conftest import try_import
from ..test_anthropic import MockAnthropic, completion_message, get_mock_chat_completion_kwargs

with try_import() as imports_successful:
    from anthropic.types.beta import (
        BetaInputJSONDelta,
        BetaMessage,
        BetaMessageDeltaUsage,
        BetaRawContentBlockDeltaEvent,
        BetaRawContentBlockStartEvent,
        BetaRawContentBlockStopEvent,
        BetaRawMessageDeltaEvent,
        BetaRawMessageStartEvent,
        BetaRawMessageStopEvent,
        BetaRawMessageStreamEvent,
        BetaTextBlock,
        BetaToolUseBlock,
        BetaUsage,
    )
    from anthropic.types.beta.beta_raw_message_delta_event import Delta

    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.models.anthropic._output_mode import has_open_object
    from pydantic_ai.providers.anthropic import AnthropicProvider

pytestmark = [pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'), pytest.mark.anyio]


class Widget(BaseModel):
    type: str
    data: dict[str, Any]


class Reply(BaseModel):
    message: str
    widgets: list[Widget]


class PlainReply(BaseModel):
    message: str


_DATA = {'url': 'https://example.com', 'label': 'Track'}


def _request(output_type: Any, content: list[Any], **model_kwargs: Any) -> tuple[Any, dict[str, Any]]:
    mock_client = MockAnthropic.create_mock(completion_message(content, BetaUsage(input_tokens=5, output_tokens=10)))
    model = AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=mock_client), **model_kwargs)
    result = Agent(model, output_type=output_type).run_sync('Send a link.')
    return result.output, get_mock_chat_completion_kwargs(mock_client)[-1]


def _shape(kwargs: dict[str, Any]) -> dict[str, Any]:
    output_config = kwargs.get('output_config')
    tools: list[dict[str, Any]] = kwargs['tools'] if isinstance(kwargs.get('tools'), list) else []
    return {
        'tool_choice': kwargs.get('tool_choice'),
        'output_format': isinstance(output_config, dict) and 'format' in output_config,
        'tools': [(tool['name'], tool.get('strict')) for tool in tools],
    }


def _tool_use_stream(message: BetaMessage) -> list[BetaRawMessageStreamEvent]:
    """The stream events of `message`, whose only block is a `tool_use`."""
    block = message.content[0]
    assert isinstance(block, BetaToolUseBlock)
    return [
        BetaRawMessageStartEvent(
            type='message_start', message=message.model_copy(update={'content': [], 'stop_reason': None})
        ),
        BetaRawContentBlockStartEvent(
            type='content_block_start', index=0, content_block=block.model_copy(update={'input': {}})
        ),
        BetaRawContentBlockDeltaEvent(
            type='content_block_delta',
            index=0,
            delta=BetaInputJSONDelta(type='input_json_delta', partial_json=json.dumps(block.input)),
        ),
        BetaRawContentBlockStopEvent(type='content_block_stop', index=0),
        BetaRawMessageDeltaEvent(
            type='message_delta',
            delta=Delta(stop_reason='tool_use'),
            usage=BetaMessageDeltaUsage(input_tokens=5, output_tokens=10),
        ),
        BetaRawMessageStopEvent(type='message_stop'),
    ]


@pytest.mark.parametrize('stream', [False, True])
async def test_dict_field_keeps_tool_output(allow_model_requests: None, stream: bool):
    """The output tool is offered unforced, as on any request that thinks, and the `dict` arrives intact."""
    tool_call = BetaToolUseBlock(
        id='toolu_1',
        name='final_result',
        input={'message': 'Here it is.', 'widgets': [{'type': 'link', 'data': _DATA}]},
        type='tool_use',
    )
    message = completion_message([tool_call], BetaUsage(input_tokens=5, output_tokens=10))
    if stream:
        mock_client = MockAnthropic.create_stream_mock(_tool_use_stream(message))
    else:
        mock_client = MockAnthropic.create_mock(message)
    agent = Agent(
        AnthropicModel('claude-haiku-5-5', provider=AnthropicProvider(anthropic_client=mock_client)),
        output_type=Reply,
    )
    if stream:
        async with agent.run_stream('Send a link.') as result:
            output = await result.get_output()
    else:
        output = (await agent.run('Send a link.')).output
    kwargs = get_mock_chat_completion_kwargs(mock_client)[-1]

    assert _shape(kwargs) == snapshot(
        {'tool_choice': {'type': 'auto'}, 'output_format': False, 'tools': [('final_result', None)]}
    )
    assert output.widgets[0].data == _DATA


def test_no_dict_field_still_uses_native_output(allow_model_requests: None):
    output, kwargs = _request(PlainReply, [BetaTextBlock(text='{"message": "Here it is."}', type='text')])

    assert _shape(kwargs)['output_format'] is True
    assert output == PlainReply(message='Here it is.')


_DICT_WARNING = '`dict` fields are not supported by Anthropic in strict mode'


def test_explicit_native_output_is_kept(allow_model_requests: None):
    """An explicit `NativeOutput` is the user's choice: it's sent as is, with the existing warning."""
    text = '{"message": "Here it is.", "widgets": [{"type": "link", "data": {}}]}'
    with pytest.warns(UserWarning, match=_DICT_WARNING):
        _, kwargs = _request(NativeOutput(Reply), [BetaTextBlock(text=text, type='text')])

    assert _shape(kwargs)['output_format'] is True


def test_native_profile_default_is_kept(allow_model_requests: None):
    text = '{"message": "Here it is.", "widgets": [{"type": "link", "data": {}}]}'
    with pytest.warns(UserWarning, match=_DICT_WARNING):
        _, kwargs = _request(
            Reply, [BetaTextBlock(text=text, type='text')], profile={'default_structured_output_mode': 'native'}
        )

    assert _shape(kwargs)['output_format'] is True


@pytest.mark.parametrize(
    'schema, expected',
    [
        ({'type': 'object', 'additionalProperties': True}, True),
        ({'type': 'object', 'additionalProperties': {'type': 'integer'}}, True),
        ({'type': 'object', 'properties': {'a': {'type': 'string'}}, 'additionalProperties': False}, False),
        ({'type': 'object', 'properties': {'a': {'type': 'string'}}}, False),
        # A property named like the keyword is a name, not the keyword.
        ({'type': 'object', 'properties': {'additionalProperties': {'type': 'object', 'properties': {}}}}, False),
        ({'$defs': {'W': {'type': 'object', 'additionalProperties': True}}, '$ref': '#/$defs/W'}, True),
        ({'type': 'array', 'items': [{'type': 'object', 'additionalProperties': True}]}, True),
    ],
)
def test_has_open_object(schema: dict[str, Any], expected: bool):
    assert has_open_object(schema) is expected
