"""Regression tests for Claude request-scoped browser page routing."""
from __future__ import annotations

import asyncio
import inspect
import unittest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.api import openai_routes
from src.api.openai_schemas import ChatCompletionRequest, ChatMessage, ResponsesRequest
from src.chatgpt.models import ChatResponse
from src.claude.client import ClaudeClient
from src.claude.selectors import ClaudeSelectors
from src.config import Config
from src.minimax.client import MiniMaxClient


class FakePool:
    def __init__(self, page):
        self.page = page

    @asynccontextmanager
    async def acquire_clean_page(self):
        yield self.page


class RecordingClient:
    def __init__(self):
        self.calls = []

    async def send_message(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return ChatResponse(message="Provider reply")


class ClaudePageRoutingTests(unittest.TestCase):
    def test_message_uses_request_page_instead_of_client_page(self):
        own_page = SimpleNamespace(url="https://claude.ai/new")
        request_page = SimpleNamespace(url="https://claude.ai/chat/1234-abcd")
        client = ClaudeClient(own_page)

        async def run():
            with (
                patch("src.claude.client.count_assistant_messages", new=AsyncMock(return_value=0)),
                patch("src.claude.client.get_latest_assistant_turn_signature", new=AsyncMock(return_value=None)),
                patch("src.claude.client.wait_for_response_complete", new=AsyncMock(return_value=True)),
                patch("src.claude.client.extract_last_response_via_copy", new=AsyncMock(return_value="Claude reply")),
                patch("src.claude.client.is_incomplete_response_text", return_value=False),
                patch("src.claude.client.random_delay", new=AsyncMock()),
                patch("src.claude.client.asyncio.sleep", new=AsyncMock()),
                patch("src.claude.client.human_type", new=AsyncMock()) as typed,
                patch.object(client, "_find_selector", new=AsyncMock(return_value="textarea")) as find,
                patch.object(client, "_upload_files", new=AsyncMock()) as upload,
                patch.object(client, "_click_send", new=AsyncMock(return_value=True)) as click,
            ):
                result = await client.send_message(
                    "Hello", file_paths=["fixture.txt"], page=request_page
                )
                typed.assert_awaited_once_with(request_page, "textarea", "Hello")
                find.assert_awaited_once_with(ClaudeSelectors.CHAT_INPUT, "chat input", request_page)
                upload.assert_awaited_once_with(["fixture.txt"], request_page)
                click.assert_awaited_once_with(request_page)
                self.assertEqual(result.message, "Claude reply")
                self.assertEqual(result.thread_id, "1234-abcd")

        asyncio.run(run())

    def test_selector_and_click_helpers_use_request_page(self):
        own_page = SimpleNamespace(wait_for_selector=AsyncMock(side_effect=AssertionError("wrong page")))
        selected = SimpleNamespace(wait_for_selector=AsyncMock(return_value=object()))
        client = ClaudeClient(own_page)

        async def run():
            result = await client._find_selector(["textarea"], "chat input", selected)
            self.assertEqual(result, "textarea")
            own_page.wait_for_selector.assert_not_awaited()
            selected.wait_for_selector.assert_awaited_once()
            with (
                patch.object(client, "_find_selector", new=AsyncMock(return_value="button")) as find,
                patch("src.claude.client.human_click", new=AsyncMock()) as clicked,
            ):
                self.assertTrue(await client._click_send(selected))
                self.assertIs(find.await_args.args[-1], selected)
                clicked.assert_awaited_once_with(selected, "button")

        asyncio.run(run())

    def test_session_records_claude_conversation_url(self):
        manager = openai_routes.PersistentSessionManager()
        page = SimpleNamespace(url="https://claude.ai/chat/1234-abcd")
        asyncio.run(manager._save_conversation_url("session-1", page))
        self.assertEqual(manager._session_urls["session-1"], "https://claude.ai/chat/1234-abcd")

    def test_chat_completions_forward_acquired_page(self):
        page, client = object(), RecordingClient()
        req = ChatCompletionRequest(model="claude-browser", messages=[ChatMessage(role="user", content="Hello")])
        with (
            patch.object(openai_routes, "_client", client),
            patch.object(openai_routes, "_page_pool", FakePool(page)),
            patch.object(Config, "PROVIDER", "claude"),
        ):
            response = asyncio.run(openai_routes.create_chat_completion(req, SimpleNamespace(headers={})))
        self.assertEqual(response.choices[0].message.content, "Provider reply")
        self.assertIs(client.calls[0][1]["page"], page)
        self.assertTrue(client.calls[0][1]["stateless"])

    def test_responses_forward_acquired_page(self):
        page, client = object(), RecordingClient()
        req = ResponsesRequest(model="claude-browser", input="Hello")
        with (
            patch.object(openai_routes, "_client", client),
            patch.object(openai_routes, "_page_pool", FakePool(page)),
            patch.object(Config, "PROVIDER", "claude"),
        ):
            response = asyncio.run(openai_routes.create_response(req, SimpleNamespace(headers={})))
        self.assertIn("output", response)
        self.assertIs(client.calls[0][1]["page"], page)

    def test_minimax_accepts_shared_provider_page_argument(self):
        self.assertIn("page", inspect.signature(MiniMaxClient.send_message).parameters)


if __name__ == "__main__":
    unittest.main()
