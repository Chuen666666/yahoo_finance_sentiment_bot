# ruff: noqa: E501 -- the embedded HTML/CSS/JavaScript is kept readable as a browser document.
from __future__ import annotations

import asyncio
import json
import threading
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .analysis import analyze_records
from .database import Database
from .entities import EntityResolver, load_manual_aliases
from .feeds import fetch_for_query
from .sentiment import ClassifierError, load_classifier
from .text import EvidenceExtractor


class RefreshPayload(BaseModel):
    query: str = Field(min_length=1, max_length=80)
    limit: int = Field(default=30, ge=1, le=100)
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class WatchPayload(BaseModel):
    query: str = Field(min_length=1, max_length=80)
    limit: int = Field(default=30, ge=1, le=100)


class WebService:
    def __init__(
        self,
        *,
        db_path: str | Path,
        aliases_path: str | Path,
        model_path: str | Path,
    ) -> None:
        self.db_path = Path(db_path)
        self.aliases_path = Path(aliases_path)
        self.model_path = Path(model_path)
        self.classifier = load_classifier(self.model_path)
        lexicon_path = self.model_path.parent / "evidence_lexicon.json"
        lexicon = json.loads(lexicon_path.read_text(encoding="utf-8")) if lexicon_path.exists() else {}
        self.extractor = EvidenceExtractor(
            {str(key): [str(value) for value in values] for key, values in lexicon.items()}
        )
        self._refresh_lock = threading.Lock()

    def articles(
        self,
        *,
        query: str = "",
        sentiment: str = "",
        limit: int = 100,
    ) -> list[dict[str, object]]:
        if sentiment not in {"", "positive", "negative", "neutral", "uncertain"}:
            raise ValueError("不支援的情緒篩選")
        with Database(self.db_path) as database:
            return database.list_analysis_rows(
                query=query,
                sentiment=sentiment,
                limit=limit,
            )

    def refresh(
        self,
        query: str,
        *,
        limit: int = 30,
        confidence_threshold: float | None = None,
    ) -> dict[str, object]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("搜尋條件不可為空白")
        with self._refresh_lock, Database(self.db_path) as database:
            entities = database.list_entities()
            if not entities:
                raise RuntimeError("公司資料庫是空的；請先執行 yfsent sync-entities")
            resolver = EntityResolver(entities, load_manual_aliases(self.aliases_path))
            target = resolver.resolve_query(normalized_query)
            records = fetch_for_query(normalized_query, target=target, limit=limit)
            database.upsert_contents(records)
            results = analyze_records(
                records,
                resolver=resolver,
                classifier=self.classifier,
                extractor=self.extractor,
                target=target,
                target_only=target is not None,
                confidence_threshold=confidence_threshold,
            )
            database.save_results(results)
        return {
            "query": normalized_query,
            "articles": len(records),
            "predictions": len(results),
            "threshold": (
                self.classifier.confidence_threshold
                if confidence_threshold is None
                else confidence_threshold
            ),
        }

    def watchlist(self) -> list[dict[str, object]]:
        with Database(self.db_path) as database:
            return database.list_watchlist()

    def add_watch(self, query: str, *, item_limit: int) -> None:
        with Database(self.db_path) as database:
            database.add_watch(query, item_limit=item_limit)

    def remove_watch(self, query: str) -> None:
        with Database(self.db_path) as database:
            database.remove_watch(query)

    def mark_watch(self, query: str, *, error: str = "") -> None:
        with Database(self.db_path) as database:
            database.update_watch_status(
                query,
                run_at=datetime.now(timezone.utc).isoformat(),
                error=error,
            )


async def _auto_refresh(service: WebService, interval_seconds: float) -> None:
    while True:
        for watch in await asyncio.to_thread(service.watchlist):
            query = str(watch["query"])
            try:
                await asyncio.to_thread(
                    service.refresh,
                    query,
                    limit=int(watch["item_limit"]),
                )
            except Exception as exc:  # scheduler must survive one failed source
                await asyncio.to_thread(service.mark_watch, query, error=str(exc))
            else:
                await asyncio.to_thread(service.mark_watch, query)
        await asyncio.sleep(interval_seconds)


def create_app(
    *,
    db_path: str | Path = Path("data/yfsent.db"),
    aliases_path: str | Path = Path("data/aliases.json"),
    model_path: str | Path = Path("artifacts/model.joblib"),
    refresh_minutes: float = 0,
) -> FastAPI:
    service = WebService(
        db_path=db_path,
        aliases_path=aliases_path,
        model_path=model_path,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = None
        if refresh_minutes > 0:
            task = asyncio.create_task(_auto_refresh(service, refresh_minutes * 60))
        yield
        if task:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(
        title="Yahoo 財經新聞情緒觀測站",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.service = service

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return INDEX_HTML

    @app.get("/api/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "model_version": service.classifier.version,
            "confidence_threshold": service.classifier.confidence_threshold,
            "auto_refresh_minutes": refresh_minutes,
        }

    @app.get("/api/articles")
    async def articles(
        query: str = Query(default="", max_length=80),
        sentiment: str = Query(default=""),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, object]:
        try:
            rows = await asyncio.to_thread(
                service.articles,
                query=query,
                sentiment=sentiment,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"count": len(rows), "items": rows}

    @app.post("/api/refresh")
    async def refresh(payload: RefreshPayload) -> dict[str, object]:
        try:
            return await asyncio.to_thread(
                service.refresh,
                payload.query,
                limit=payload.limit,
                confidence_threshold=payload.confidence_threshold,
            )
        except (ClassifierError, RuntimeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/watchlist")
    async def watchlist() -> dict[str, object]:
        rows = await asyncio.to_thread(service.watchlist)
        return {"count": len(rows), "items": rows}

    @app.post("/api/watchlist")
    async def add_watch(payload: WatchPayload) -> dict[str, str]:
        try:
            await asyncio.to_thread(
                service.add_watch,
                payload.query,
                item_limit=payload.limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "ok", "query": payload.query.strip()}

    @app.delete("/api/watchlist")
    async def remove_watch(query: str = Query(min_length=1, max_length=80)) -> dict[str, str]:
        await asyncio.to_thread(service.remove_watch, query)
        return {"status": "ok", "query": query.strip()}

    return app


INDEX_HTML = r"""<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>市場情緒觀測站</title>
  <style>
    :root { color-scheme: light; --ink:#17221c; --muted:#647269; --paper:#f3f1e9;
      --card:#fffdf7; --line:#d8d7ce; --accent:#165d42; --pos:#207a50;
      --neg:#b43c35; --neu:#59646b; --unc:#a66a12; }
    * { box-sizing: border-box; }
    body { margin:0; background:var(--paper); color:var(--ink); font-family:ui-sans-serif,
      system-ui,"Noto Sans TC",sans-serif; }
    header { padding:64px 24px 36px; background:#173e31; color:#f8f3df; }
    header div, main { max-width:1040px; margin:auto; }
    h1 { margin:0 0 10px; font-family:ui-serif,"Noto Serif TC",serif; font-size:clamp(2rem,5vw,4rem); }
    header p { margin:0; color:#d4dfd5; }
    main { padding:28px 20px 64px; }
    .panel { display:grid; grid-template-columns:minmax(220px,1fr) 170px auto auto;
      gap:10px; padding:16px; background:var(--card); border:1px solid var(--line); border-radius:16px; }
    input, select, button { min-height:44px; border:1px solid var(--line); border-radius:10px;
      padding:0 13px; font:inherit; background:white; }
    button { border-color:var(--accent); background:var(--accent); color:white; cursor:pointer; font-weight:700; }
    button.secondary { background:white; color:var(--accent); }
    button:disabled { opacity:.55; cursor:wait; }
    .status { min-height:28px; margin:12px 2px; color:var(--muted); }
    .watchbar { display:flex; flex-wrap:wrap; align-items:center; gap:8px; margin:16px 0 26px; }
    .watch { display:inline-flex; align-items:center; gap:8px; background:#e5ebe4; padding:7px 10px;
      border-radius:999px; font-size:.9rem; }
    .watch button { min-height:auto; padding:0; border:0; color:#79342e; background:transparent; }
    .feed { display:grid; gap:14px; }
    article { padding:20px; background:var(--card); border:1px solid var(--line); border-radius:16px; }
    .meta { display:flex; flex-wrap:wrap; align-items:center; gap:8px; color:var(--muted); font-size:.86rem; }
    .badge { padding:4px 9px; border-radius:999px; color:white; font-weight:800; letter-spacing:.02em; }
    .positive { background:var(--pos); } .negative { background:var(--neg); }
    .neutral { background:var(--neu); } .uncertain { background:var(--unc); }
    h2 { margin:12px 0 8px; font-size:1.2rem; line-height:1.45; }
    h2 a { color:var(--ink); text-decoration:none; } h2 a:hover { color:var(--accent); }
    .summary { color:#435047; line-height:1.65; }
    .evidence { margin-top:12px; padding:10px 12px; border-left:3px solid var(--accent);
      background:#edf2ec; color:#34483d; }
    .meter { width:100px; height:7px; overflow:hidden; background:#deded7; border-radius:20px; }
    .meter i { display:block; height:100%; background:var(--accent); }
    .empty { padding:48px 20px; text-align:center; color:var(--muted); }
    @media (max-width:760px) { .panel { grid-template-columns:1fr 1fr; }
      .panel input { grid-column:1/-1; } }
  </style>
</head>
<body>
  <header><div><h1>市場情緒觀測站</h1><p>Yahoo 股市 RSS × 公司目標式情緒分析</p></div></header>
  <main>
    <section class="panel" aria-label="新聞搜尋與篩選">
      <input id="query" placeholder="公司名稱、股票代碼或關鍵字" autocomplete="off">
      <select id="sentiment" aria-label="情緒篩選">
        <option value="">全部情緒</option><option value="positive">正面</option>
        <option value="negative">負面</option><option value="neutral">中立</option>
        <option value="uncertain">待確認</option>
      </select>
      <button id="search" class="secondary">搜尋資料庫</button>
      <button id="refresh">抓取最新新聞</button>
    </section>
    <div id="status" class="status" role="status"></div>
    <div class="watchbar"><strong>自動追蹤：</strong><span id="watchlist"></span>
      <button id="addWatch" class="secondary">＋ 追蹤目前條件</button></div>
    <section id="feed" class="feed" aria-live="polite"></section>
  </main>
  <script>
    const $ = id => document.getElementById(id);
    const labels = {positive:'正面', negative:'負面', neutral:'中立', uncertain:'待確認'};
    const escapeHtml = value => String(value ?? '').replace(/[&<>'"]/g, char =>
      ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[char]));
    const safeUrl = value => /^https?:\/\//i.test(value || '') ? escapeHtml(value) : '#';
    const setBusy = busy => { $('refresh').disabled = busy; $('search').disabled = busy; };
    async function api(url, options) {
      const response = await fetch(url, options);
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || '請求失敗');
      return data;
    }
    function render(items) {
      if (!items.length) { $('feed').innerHTML = '<div class="empty">目前沒有符合條件的新聞。</div>'; return; }
      $('feed').innerHTML = items.map(item => {
        const evidence = item.evidence?.[0]?.text;
        const confidence = Math.round(Number(item.confidence) * 100);
        return `<article><div class="meta"><span class="badge ${escapeHtml(item.sentiment)}">${labels[item.sentiment] || escapeHtml(item.sentiment)}</span>
          <strong>${escapeHtml(item.entity_name)} (${escapeHtml(item.entity_code)})</strong>
          <span>${escapeHtml((item.published_at || '').replace('T',' ').slice(0,16))}</span>
          <span class="meter" title="信心 ${confidence}%"><i style="width:${confidence}%"></i></span><span>${confidence}%</span></div>
          <h2><a href="${safeUrl(item.url)}" target="_blank" rel="noopener noreferrer">${escapeHtml(item.title)}</a></h2>
          <div class="summary">${escapeHtml(item.summary)}</div>
          ${evidence ? `<div class="evidence">判斷依據：${escapeHtml(evidence)}</div>` : ''}</article>`;
      }).join('');
    }
    async function loadArticles() {
      setBusy(true);
      try {
        const params = new URLSearchParams({query:$('query').value.trim(), sentiment:$('sentiment').value, limit:'100'});
        const data = await api('/api/articles?' + params);
        render(data.items); $('status').textContent = `找到 ${data.count} 筆分析結果`;
      } catch (error) { $('status').textContent = error.message; } finally { setBusy(false); }
    }
    async function refresh() {
      const query = $('query').value.trim();
      if (!query) { $('status').textContent = '請先輸入公司名稱、股票代碼或關鍵字。'; return; }
      setBusy(true); $('status').textContent = '正在取得 RSS 並分析，請稍候…';
      try {
        const data = await api('/api/refresh', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query,limit:30})});
        $('status').textContent = `已取得 ${data.articles} 篇，產生 ${data.predictions} 筆公司情緒分析。`;
        await loadArticles();
      } catch (error) { $('status').textContent = error.message; } finally { setBusy(false); }
    }
    async function loadWatchlist() {
      const data = await api('/api/watchlist');
      $('watchlist').innerHTML = data.items.length ? data.items.map(item =>
        `<span class="watch">${escapeHtml(item.query)}<button data-query="${escapeHtml(item.query)}" title="停止追蹤">×</button></span>`).join('') : '<span>尚未設定</span>';
      document.querySelectorAll('.watch button').forEach(button => button.onclick = async () => {
        await api('/api/watchlist?query=' + encodeURIComponent(button.dataset.query), {method:'DELETE'}); await loadWatchlist();
      });
    }
    $('search').onclick = loadArticles; $('refresh').onclick = refresh;
    $('sentiment').onchange = loadArticles; $('query').onkeydown = event => { if (event.key === 'Enter') loadArticles(); };
    $('addWatch').onclick = async () => {
      const query = $('query').value.trim(); if (!query) { $('status').textContent = '請先輸入追蹤條件。'; return; }
      await api('/api/watchlist', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({query,limit:30})});
      $('status').textContent = `已加入自動追蹤：${query}`; await loadWatchlist();
    };
    loadWatchlist().catch(error => $('status').textContent = error.message); loadArticles();
  </script>
</body>
</html>"""
