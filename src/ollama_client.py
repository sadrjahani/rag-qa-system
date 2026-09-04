"""Small standard-library client for a local Ollama server."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """Raised when Ollama cannot generate a response."""


class OllamaClient:
    """Call Ollama's local ``/api/generate`` endpoint without extra packages."""

    def __init__(self, model: str, base_url: str = "http://localhost:11434", timeout: int = 120) -> None:
        if not model.strip():
            raise ValueError("Ollama model must not be empty.")
        self.model = model
        self.url = base_url.rstrip("/") + "/api/generate"
        self.timeout = timeout

    def generate(self, prompt: str, *, temperature: float = 0.0) -> str:
        payload = json.dumps({
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        }).encode("utf-8")
        request = Request(
            self.url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise OllamaError(f"Ollama request failed at {self.url}: {exc}") from exc

        text = body.get("response")
        if not isinstance(text, str) or not text.strip():
            raise OllamaError("Ollama returned no response text.")
        return text.strip()