import importlib.util
import unittest
from unittest.mock import patch

WEB_AVAILABLE = bool(importlib.util.find_spec("fastapi") and importlib.util.find_spec("httpx"))

if WEB_AVAILABLE:
    import httpx

    from yfsent.web import create_app


class FakeClassifier:
    version = "fake-v1"
    confidence_threshold = 0.42


class FakeWebService:
    def __init__(self) -> None:
        self.classifier = FakeClassifier()
        self.watches: list[dict[str, object]] = []

    def articles(self, **_: object) -> list[dict[str, object]]:
        return [
            {
                "id": "item-1",
                "title": "台積電營收成長",
                "summary": "測試摘要",
                "url": "https://example.test/1",
                "source": "測試來源",
                "published_at": "2026-09-22T01:00:00+00:00",
                "entity_code": "2330",
                "entity_name": "台積電",
                "sentiment": "positive",
                "raw_label": "positive",
                "confidence": 0.81,
                "review_required": False,
                "target_mentioned": True,
                "evidence": [],
                "model_version": "fake-v1",
                "analyzed_at": "2026-09-22T01:01:00+00:00",
            }
        ]

    def refresh(self, query: str, **_: object) -> dict[str, object]:
        return {"query": query, "articles": 1, "predictions": 1, "threshold": 0.42}

    def watchlist(self) -> list[dict[str, object]]:
        return self.watches

    def add_watch(self, query: str, *, item_limit: int) -> None:
        self.watches = [{"query": query, "item_limit": item_limit, "enabled": True}]

    def remove_watch(self, query: str) -> None:
        self.watches = [item for item in self.watches if item["query"] != query]


@unittest.skipUnless(WEB_AVAILABLE, "FastAPI/httpx web dependencies are not installed")
class WebTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.service = FakeWebService()
        self.patcher = patch("yfsent.web.WebService", return_value=self.service)
        self.patcher.start()
        transport = httpx.ASGITransport(app=create_app(refresh_minutes=0))
        self.client = httpx.AsyncClient(transport=transport, base_url="http://test")

    async def asyncTearDown(self) -> None:
        await self.client.aclose()
        self.patcher.stop()

    async def test_home_health_and_article_filter(self) -> None:
        response = await self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("市場情緒觀測站", response.text)

        health = (await self.client.get("/api/health")).json()
        self.assertEqual(health["model_version"], "fake-v1")
        articles = (
            await self.client.get("/api/articles", params={"sentiment": "positive"})
        ).json()
        self.assertEqual(articles["count"], 1)
        self.assertEqual(articles["items"][0]["entity_code"], "2330")

    async def test_refresh_and_watchlist(self) -> None:
        refreshed = await self.client.post(
            "/api/refresh", json={"query": "台積電", "limit": 10}
        )
        self.assertEqual(refreshed.status_code, 200)
        self.assertEqual(refreshed.json()["predictions"], 1)

        added = await self.client.post(
            "/api/watchlist", json={"query": "台積電", "limit": 20}
        )
        self.assertEqual(added.status_code, 200)
        self.assertEqual((await self.client.get("/api/watchlist")).json()["count"], 1)
        removed = await self.client.delete("/api/watchlist", params={"query": "台積電"})
        self.assertEqual(removed.status_code, 200)
        self.assertEqual((await self.client.get("/api/watchlist")).json()["count"], 0)


if __name__ == "__main__":
    unittest.main()
