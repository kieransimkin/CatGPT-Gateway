"""
FastAPI server — serves ChatGPT as an API.

Launches the browser on startup, shuts it down on exit.

Usage:
    python -m src.api.server
    # or
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from src.browser.manager import BrowserManager
from src.browser.auto_login import ensure_logged_in
from src.chatgpt.client import ChatGPTClient
from src.claude.client import ClaudeClient
from src.minimax.client import MiniMaxClient
from src.config import Config
from src.api.routes import router, set_client
from src.api.openai_routes import openai_router, set_openai_clients
from src.log import setup_logging

log = setup_logging("api_server")

# Global instances — needed for lifespan
_browser: BrowserManager | None = None
_client: ChatGPTClient | ClaudeClient | MiniMaxClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run every enabled model concurrently with isolated provider tabs."""
    global _browser, _client
    Config.validate_models()
    model_ids = Config.available_models()
    clients: dict[str, ChatGPTClient | ClaudeClient | MiniMaxClient] = {}
    browser_models = [m for m in model_ids if Config.uses_browser(Config.model_provider(m))]
    _browser = None
    _client = None

    try:
        if browser_models:
            _browser = BrowserManager()
            initial_page = await _browser.start()
            for index, model_id in enumerate(browser_models):
                provider = Config.model_provider(model_id)
                target_url = Config.provider_url(provider)
                page = initial_page if index == 0 else await _browser.new_page()
                for attempt in range(1, 6):
                    try:
                        await page.goto(target_url, wait_until="domcontentloaded", timeout=25000)
                        break
                    except Exception as exc:
                        if attempt == 5:
                            raise
                        log.warning("%s navigation retry %d: %s", model_id, attempt, exc)
                        await asyncio.sleep(5 * attempt)

                if index == 0:
                    # Stealth patches are deliberately applied after navigation.
                    await _browser.apply_stealth_patches()
                await asyncio.sleep(1)
                if not await _browser.is_logged_in(provider=provider, page=page):
                    logged_in = await ensure_logged_in(_browser, provider=provider, page=page)
                    if not logged_in:
                        raise RuntimeError(f"Login was not verified for {model_id}")
                clients[model_id] = (
                    ClaudeClient(page) if provider == "claude" else ChatGPTClient(page)
                )
                log.info("Browser model ready: %s (%s)", model_id, target_url)

        for model_id in model_ids:
            if Config.model_provider(model_id) == "minimax":
                clients[model_id] = MiniMaxClient(model=model_id)
                log.info("API model ready: %s", model_id)
        if "MiniMax-M2.7" in Config.ENABLED_MODELS and not Config.MINIMAX_API_KEY:
            log.warning("MiniMax-M2.7 not advertised: set MINIMAX_API_KEY to enable")

        _client = clients[Config.DEFAULT_MODEL]
        set_client(_client, _browser)
        set_openai_clients(clients, _browser)
        log.info("Gateway ready; models: %s", ", ".join(clients))
        yield
    finally:
        if _browser is not None:
            await _browser.close()
        _browser = None
        _client = None


app = FastAPI(
    title="CatGPT Gateway API",
    description="Provider gateway with OpenAI-compatible routes.",
    version="1.0.0",
    lifespan=lifespan,
)

# ── Bearer Token Auth Middleware ────────────────────────────────
class BearerTokenMiddleware:
    """
    Pure ASGI middleware for Bearer token auth.

    Uses raw ASGI protocol instead of BaseHTTPMiddleware to avoid the
    Python 3.9 event-loop mismatch bug that corrupts asyncio.Lock
    when exceptions propagate through BaseHTTPMiddleware's task group.

    Skips auth for /docs, /openapi.json, and health-check paths.
    """

    OPEN_PATHS = {b"/docs", b"/redoc", b"/openapi.json", b"/healthz", b"/preview", b"/v1/preview", b"/v1/screenshot.png"}

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        token = Config.API_TOKEN
        if not token:
            await self.app(scope, receive, send)
            return

        path_str = (scope.get("path") or "").rstrip("/")
        if (
            path_str in {"", "/docs", "/redoc", "/openapi.json", "/healthz", "/preview", "/v1/preview", "/v1/screenshot.png", "/v1/tabs", "/favicon.ico"}
            or path_str.startswith("/preview")
            or path_str.startswith("/v1/screenshot")
            or path_str.startswith("/v1/tabs")
        ):
            await self.app(scope, receive, send)
            return

        # Extract Authorization or x-api-key header
        headers = dict(scope.get("headers", []))
        auth_value = headers.get(b"authorization", b"").decode()

        provided = ""
        if auth_value.startswith("Bearer "):
            provided = auth_value[7:]
        elif b"x-api-key" in headers:
            provided = headers[b"x-api-key"].decode()
        elif b"anthropic-api-key" in headers:
            provided = headers[b"anthropic-api-key"].decode()

        if provided != token:
            response = JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Invalid or missing API token. Set Authorization: Bearer <API_TOKEN>",
                        "type": "auth_error",
                    }
                },
            )
            await response(scope, receive, send)
            return

        await self.app(scope, receive, send)


app.add_middleware(BearerTokenMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(openai_router)


@app.get("/healthz", include_in_schema=False)
async def healthz():
    """Unauthenticated health-check for Docker / load-balancers."""
    return {"status": "ok"}


def _ensure_xvfb() -> None:
    """
    On Linux, automatically re-exec the process under xvfb-run if:
      - We are not already inside a virtual display (DISPLAY != :99xx)
      - xvfb-run is available
      - USE_XVFB env var is not explicitly set to 'false'

    This guarantees zero focus-stealing even when the server is started
    directly (e.g. by a code agent via `python -m src.api.server`) rather
    than through start_server.sh.
    """
    if sys.platform != "linux":
        return
    if os.environ.get("USE_XVFB", "").lower() == "false":
        return
    if os.environ.get("_XVFB_WRAPPED", "") == "1":
        return  # already inside xvfb-run — avoid infinite loop

    display = os.environ.get("DISPLAY", "")
    # If already on a virtual display (:99, :100, etc.) skip re-wrapping
    if display and ":99" in display:
        return

    xvfb = shutil.which("xvfb-run")
    if not xvfb:
        return

    log.info(
        "Auto-detected: not running under Xvfb virtual display. "
        "Re-launching inside Xvfb to prevent focus interruption..."
    )
    env = os.environ.copy()
    env["_XVFB_WRAPPED"] = "1"
    cmd = [xvfb, "-a", sys.executable, "-m", "src.api.server"] + sys.argv[1:]
    result = subprocess.run(cmd, env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    import uvicorn

    _ensure_xvfb()

    uvicorn.run(
        "src.api.server:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        log_level="info",
    )
