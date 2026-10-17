import asyncio
import json
import unittest
from unittest.mock import patch

import httpx

import retrieve as example


class RetrievalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.requests = []
        self.payload = {
            "search_id": "search_fixture",
            "session_id": "session_fixture",
            "results": [
                {"url": "https://docs.langchain.com/retrieval", "title": "Retrieval",
                 "excerpts": ["Retrieve context for a question.", "Use it in a prompt."]},
                {"url": "https://example.org/empty", "excerpts": []},
            ],
        }
        self.tool_error = False
        self.structured = True
        self.wait_for_call = None
        self.cancelled = False
        self.original_client = httpx.AsyncClient
        self.clients = []

    async def handle(self, request):
        self.requests.append(request)
        message = json.loads(request.content)
        if "id" not in message:
            return httpx.Response(202)
        if message["method"] == "initialize":
            result = {
                "protocolVersion": message["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "offline-fixture", "version": "1"},
            }
        elif message["method"] == "tools/call":
            if self.wait_for_call:
                self.wait_for_call.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.cancelled = True
            result = {
                "content": [{"type": "text", "text": json.dumps(self.payload)}],
                "isError": self.tool_error,
            }
            if self.structured:
                result["structuredContent"] = self.payload
        else:
            self.fail(f"Unexpected MCP method: {message['method']}")
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def client(self, **kwargs):
        client = self.original_client(transport=httpx.MockTransport(self.handle), **kwargs)
        self.clients.append(client)
        return client

    async def run_retrieve(self, **kwargs):
        with patch.object(example.httpx, "AsyncClient", side_effect=self.client):
            return await example.retrieve("How does RAG work?", ["LangChain RAG documentation"], **kwargs)

    async def test_anonymous_request_documents_and_prompt(self):
        with patch.dict("os.environ", {"PARALLEL_API_KEY": "unused", "OPENAI_API_KEY": "unused"}):
            docs = await self.run_retrieve()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0].page_content, "Retrieve context for a question.\n\nUse it in a prompt.")
        self.assertEqual(docs[0].metadata["source"], "https://docs.langchain.com/retrieval")
        messages = example.build_prompt("How does RAG work?", docs).to_messages()
        self.assertIn("cite source numbers", messages[0].content)
        self.assertIn("[1] Retrieval\nSource: https://docs.langchain.com/retrieval", messages[1].content)
        self.assertIn(docs[0].page_content, messages[1].content)
        self.assertIn("Question: How does RAG work?", messages[1].content)
        methods = [json.loads(r.content)["method"] for r in self.requests]
        self.assertEqual(methods, ["initialize", "notifications/initialized", "tools/call"])
        for request in self.requests:
            self.assertEqual(str(request.url), "https://search.parallel.ai/mcp")
            self.assertEqual(request.headers["user-agent"], example.USER_AGENT)
            self.assertNotIn("authorization", request.headers)
        call = json.loads(self.requests[-1].content)["params"]
        self.assertEqual(call["name"], "web_search")
        self.assertEqual(call["arguments"]["objective"], "How does RAG work?")
        self.assertEqual(call["arguments"]["search_queries"], ["LangChain RAG documentation"])
        self.assertEqual(len(call["arguments"]["session_id"]), 32)
        self.assertTrue(self.clients[0].is_closed)

    async def test_json_text_fallback_and_missing_title(self):
        self.structured = False
        self.payload["results"][0]["title"] = None
        docs = await self.run_retrieve()
        self.assertEqual(docs[0].metadata["title"], docs[0].metadata["source"])

    async def test_empty_search(self):
        self.payload["results"] = []
        self.assertEqual(await self.run_retrieve(), [])

    async def test_tool_error_is_not_an_empty_search(self):
        self.tool_error = True
        self.payload = {"error": "Free anonymous quota exceeded"}
        with self.assertRaisesRegex(RuntimeError, "Free anonymous quota exceeded"):
            await self.run_retrieve()

    async def test_malformed_response(self):
        for payload in ({}, {"results": None}, {"results": [{"url": "bad", "excerpts": [42]}]}):
            with self.subTest(payload=payload):
                self.payload = payload
                with self.assertRaisesRegex(RuntimeError, "Unexpected web_search response"):
                    await self.run_retrieve()

    async def test_timeout_cancels_request_and_closes_client(self):
        self.wait_for_call = asyncio.Event()
        with self.assertRaises(TimeoutError):
            await self.run_retrieve(timeout=0.1)
        self.assertTrue(self.wait_for_call.is_set())
        self.assertTrue(self.cancelled)
        self.assertTrue(self.clients[0].is_closed)

    async def test_cancellation_propagates_and_closes_client(self):
        self.wait_for_call = asyncio.Event()
        task = asyncio.create_task(self.run_retrieve())
        await asyncio.wait_for(self.wait_for_call.wait(), timeout=5)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(self.cancelled)
        self.assertTrue(self.clients[0].is_closed)

    async def test_invalid_inputs_do_not_connect(self):
        for question, queries in ((" ", ["query"]), ("question", []), ("question", [" "])):
            with self.subTest(question=question, queries=queries):
                with patch.object(example.httpx, "AsyncClient") as client:
                    with self.assertRaises(ValueError):
                        await example.retrieve(question, queries)
                    client.assert_not_called()


if __name__ == "__main__":
    unittest.main()
