"""Shared LLM client (OpenAI Responses API / Google Gemini) used by every language-model agent."""

from __future__ import annotations

import json
import os
from typing import Optional, Type

from dotenv import load_dotenv
from pydantic import BaseModel

# Load OPENAI_API_KEY / GEMINI_API_KEY from a .env file in the working
# directory (or any parent), if present. override=True so a stale key left
# exported in the shell from an earlier session can't shadow a rotated key
# in .env.
load_dotenv(override=True)


# DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_MODEL = "gpt-4o-mini"


def _infer_provider(model_name: str) -> str:
    """Pick the backing API for a model name.

    Every agent already threads ``model_name`` through end to end, so the
    provider is inferred from its prefix instead of adding a second
    parameter that would need to be plumbed through the same call sites.
    """
    lowered = model_name.lower()
    if lowered.startswith("gemini"):
        return "gemini"
    return "openai"


def _pydantic_schema_for_gemini(response_model: Type[BaseModel]) -> dict:
    """Convert a pydantic model's JSON schema into Gemini's accepted subset.

    Gemini's ``response_schema`` rejects any dict containing an
    ``additionalProperties`` key outright (400 "Unknown name
    'additional_properties'") -- passing the pydantic *class* straight to the
    SDK triggers this, because its own model-to-schema conversion carries
    that key through from any model using ``ConfigDict(extra="forbid")``.
    Gemini's schema is closed-by-default (no arbitrary extra keys) without
    the flag, so it's safe to just strip it recursively rather than maintain
    a second, Gemini-specific set of models.
    """

    def _strip(node):
        if isinstance(node, dict):
            return {k: _strip(v) for k, v in node.items() if k != "additionalProperties"}
        if isinstance(node, list):
            return [_strip(v) for v in node]
        return node

    return _strip(response_model.model_json_schema())


def _repair_bare_latex_backslashes(text: str) -> str:
    """Double a bare backslash so LLM-authored LaTeX survives ``json.loads``.

    Models reliably emit a single backslash for LaTeX commands (``\\textbf``,
    ``\\ref``, ``\\Cref``, ``\\emph`` ...) inside a JSON string value instead of
    the ``\\\\`` JSON requires. For a command letter that is also a valid
    single-character JSON escape (``b``/``f``/``n``/``r``/``t``/``u``),
    ``json.loads`` does not raise -- it silently turns ``"\\textbf{...}"`` into
    a literal TAB character followed by the text ``extbf{...}``. For any other
    letter (``\\emph``, ``\\Cref`` ...) it raises "Invalid \\escape" partway
    through the string, which breaks parsing of the *entire* JSON payload, not
    just that one field.

    Every agent's JSON payload in this pipeline is LaTeX-bearing content with
    no legitimate use for a raw JSON control-character escape, so any
    backslash that is not already the second half of a ``\\\\``/``\\"``/``\\/``
    pair, and is not a genuine ``\\uXXXX`` unicode escape, is treated as the
    start of a LaTeX command and doubled so it survives parsing intact.
    """
    output: list[str] = []
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char != "\\" or index + 1 >= length:
            output.append(char)
            index += 1
            continue

        nxt = text[index + 1]
        if nxt in ("\\", '"', "/"):
            output.append(char)
            output.append(nxt)
            index += 2
            continue
        if nxt == "u" and index + 6 <= length and all(
            c in "0123456789abcdefABCDEF" for c in text[index + 2 : index + 6]
        ):
            output.append(text[index : index + 6])
            index += 6
            continue

        output.append("\\\\")
        output.append(nxt)
        index += 2
    return "".join(output)


class LLMClient:
    """Stateless wrapper around the OpenAI Responses API or Google Gemini API.

    It preserves the original ``generate`` interface used by the existing
    agents, while moving inference off the local GPU. No tokenizer, checkpoint,
    CUDA device, or quantization configuration is loaded by this client. The
    backing provider is picked from ``model_name`` (see ``_infer_provider``),
    so callers select a model, not an SDK.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        api_key: Optional[str] = None,
        # Reasoning-family models (gpt-5, o-series, ...) routinely take over
        # a minute per call -- a 300s ceiling gives them room without
        # papering over a genuinely stuck request forever.
        timeout: float = 300.0,
        max_retries: int = 2,
        max_new_tokens: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.provider = _infer_provider(model_name)

        if self.provider == "gemini":
            self._init_gemini(api_key=api_key)
        else:
            self._init_openai(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def _init_openai(self, *, api_key: Optional[str], timeout: float, max_retries: int) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on the deployment environment
            raise ImportError(
                "The OpenAI SDK is required. Install the project dependencies with "
                "`python3 -m pip install -r requirements.txt`."
            ) from exc

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export an OpenAI API key before "
                "starting the document-generation pipeline."
            )
        self.client = OpenAI(api_key=api_key, timeout=timeout, max_retries=max_retries)

    def _init_gemini(self, *, api_key: Optional[str]) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - depends on the deployment environment
            raise ImportError(
                "The google-genai SDK is required for Gemini models. Install the "
                "project dependencies with `python3 -m pip install -r requirements.txt`."
            ) from exc

        api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export a Gemini API key before "
                "starting the document-generation pipeline."
            )
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate text while retaining the old local-client method shape."""
        target_max_tokens = max_new_tokens or self.max_new_tokens
        if target_max_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")

        if self.provider == "gemini":
            return self._generate_gemini(system_prompt, user_prompt, temperature, top_p, target_max_tokens)
        return self._generate_openai(system_prompt, user_prompt, temperature, top_p, target_max_tokens)

    def supports_structured_output(self) -> bool:
        """Whether ``generate_structured`` has a native implementation for this provider.

        Only OpenAI and Gemini have one today. Any other provider (a local/
        OpenAI-compatible endpoint such as Qwen behind vLLM, for example)
        isn't guaranteed to honor a strict JSON schema the same way, so
        callers should fall back to prompt-only ``generate`` instead of
        assuming this works everywhere.
        """
        return self.provider in ("openai", "gemini")

    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        *,
        schema_name: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        max_new_tokens: Optional[int] = None,
    ) -> str:
        """Generate text that is guaranteed (by the provider) to validate against ``response_model``.

        Returns the raw JSON text -- callers still parse/validate it
        themselves, same as ``generate`` + ``extract_json_block``. Only
        implemented for providers where the API itself enforces the schema
        (see ``supports_structured_output``); call ``generate`` instead for
        anything else.
        """
        if not self.supports_structured_output():
            raise NotImplementedError(
                f"generate_structured() has no implementation for provider '{self.provider}'; "
                "use generate() and validate the parsed JSON yourself."
            )

        target_max_tokens = max_new_tokens or self.max_new_tokens
        if target_max_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero.")

        if self.provider == "gemini":
            return self._generate_gemini_structured(
                system_prompt, user_prompt, response_model, temperature, top_p, target_max_tokens
            )
        return self._generate_openai_structured(
            system_prompt, user_prompt, response_model, schema_name, temperature, top_p, target_max_tokens
        )

    def _generate_openai_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        schema_name: str,
        temperature: float,
        top_p: float,
        target_max_tokens: int,
    ) -> str:
        response = self._create_openai_response({
            "model": self.model_name,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": target_max_tokens,
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": response_model.model_json_schema(),
                    "strict": True,
                }
            },
        })
        return self._text_from_openai_response(response)

    def _generate_gemini_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[BaseModel],
        temperature: float,
        top_p: float,
        target_max_tokens: int,
    ) -> str:
        from google.genai import types
        from google.genai.errors import APIError

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=target_max_tokens,
                    response_mime_type="application/json",
                    response_schema=_pydantic_schema_for_gemini(response_model),
                ),
            )
            if response.candidates and response.candidates[0].finish_reason == "MAX_TOKENS":
                raise RuntimeError(
                    "Gemini response was truncated by max_output_tokens; increase the token budget."
                )
        except APIError as exc:
            raise RuntimeError(f"Gemini API returned an error: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned no text output for this request.")
        return text

    def _create_openai_response(self, request_kwargs: dict):
        """Call ``responses.create``, dropping any parameter the API rejects.

        Reasoning-family models (o-series, gpt-5, ...) reject sampling
        parameters like temperature/top_p outright with a 400 rather than
        silently ignoring them, and which parameters are rejected varies by
        model and changes over time. Hardcoding a model-name allowlist would
        go stale, so instead: drop whichever parameter the API says it
        doesn't support and retry, until the call succeeds or a rejection
        can't be resolved by dropping a parameter.
        """
        from openai import APIConnectionError, APIStatusError, APITimeoutError

        droppable_params = {"temperature", "top_p"}
        request_kwargs = dict(request_kwargs)
        while True:
            try:
                return self.client.responses.create(**request_kwargs)
            except APIStatusError as exc:
                rejected_param = None
                if exc.status_code == 400 and isinstance(exc.body, dict):
                    # The SDK already unwraps the response's top-level "error"
                    # object into exc.body, so the key is directly on it.
                    rejected_param = exc.body.get("param")
                if rejected_param in droppable_params and rejected_param in request_kwargs:
                    request_kwargs.pop(rejected_param)
                    continue
                raise RuntimeError(
                    f"OpenAI API returned HTTP {exc.status_code}: {exc.message}"
                ) from exc
            except APITimeoutError as exc:
                raise RuntimeError("OpenAI request timed out after configured retries.") from exc
            except APIConnectionError as exc:
                raise RuntimeError("Could not connect to the OpenAI API.") from exc

    @staticmethod
    def _text_from_openai_response(response) -> str:
        if response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens":
            raise RuntimeError(
                "OpenAI response was truncated by max_output_tokens; increase the token budget."
            )
        text = response.output_text.strip()
        if not text:
            raise RuntimeError("OpenAI returned no text output for this request.")
        return text

    def _generate_openai(
        self, system_prompt: str, user_prompt: str, temperature: float, top_p: float, target_max_tokens: int
    ) -> str:
        response = self._create_openai_response({
            "model": self.model_name,
            "instructions": system_prompt,
            "input": user_prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": target_max_tokens,
            # Do not retain generated documents as server-side API state.
            "store": False,
        })
        return self._text_from_openai_response(response)

    def _generate_gemini(
        self, system_prompt: str, user_prompt: str, temperature: float, top_p: float, target_max_tokens: int
    ) -> str:
        from google.genai import types
        from google.genai.errors import APIError

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=temperature,
                    top_p=top_p,
                    max_output_tokens=target_max_tokens,
                ),
            )
            if response.candidates and response.candidates[0].finish_reason == "MAX_TOKENS":
                raise RuntimeError(
                    "Gemini response was truncated by max_output_tokens; increase the token budget."
                )
        except APIError as exc:
            raise RuntimeError(f"Gemini API returned an error: {exc}") from exc

        text = (response.text or "").strip()
        if not text:
            raise RuntimeError("Gemini returned no text output for this request.")
        return text

    @staticmethod
    def extract_json_block(text: str) -> str:
        """Extract the first valid JSON object or array from a model response."""
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
            stripped = stripped.rsplit("```", 1)[0].strip()

        stripped = _repair_bare_latex_backslashes(stripped)
        decoder = json.JSONDecoder()
        for index, char in enumerate(stripped):
            if char not in "{[":
                continue
            try:
                value, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            return json.dumps(value, ensure_ascii=False)
        return stripped
