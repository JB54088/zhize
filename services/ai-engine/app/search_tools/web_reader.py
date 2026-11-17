"""Bounded public web and PDF reader with SSRF and robots.txt protections."""

from __future__ import annotations

import asyncio
import ipaddress
import io
import socket
from collections.abc import Callable, Sequence
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx


MAX_PAGE_BYTES = 5 * 1024 * 1024
MAX_ROBOTS_BYTES = 256 * 1024
MAX_EXTRACTED_CHARS = 40_000
MAX_PDF_PAGES = 40
MAX_REDIRECTS = 4
DEFAULT_TIMEOUT_SECONDS = 12.0
USER_AGENT = "ZhizeInfoBot/0.4 (+public evidence fetch; respects robots.txt)"


class WebReadError(Exception):
    """A safe-to-report webpage read failure with a stable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class _VisibleTextParser(HTMLParser):
    _HIDDEN = {"script", "style", "noscript", "svg", "template", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self.description = ""
        self._hidden_depth = 0
        self._title_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attrs_map = dict(attrs)
        if tag in self._HIDDEN:
            self._hidden_depth += 1
        if tag == "title":
            self._title_depth += 1
        if tag == "meta" and attrs_map.get("name", "").lower() in {"description", "og:description"}:
            content = attrs_map.get("content")
            if content and not self.description:
                self.description = content.strip()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        if tag in self._HIDDEN and self._hidden_depth:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self._title_depth:
            self.title_parts.append(text)
        if not self._hidden_depth:
            self.parts.append(text)


class SafeWebReader:
    """Read publicly accessible HTML/PDF with bounded requests and extraction."""

    def __init__(
        self,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_page_bytes: int = MAX_PAGE_BYTES,
        max_redirects: int = MAX_REDIRECTS,
        resolve_host: Callable[[str], Sequence[str]] | None = None,
    ) -> None:
        self.transport = transport
        self.timeout_seconds = min(max(float(timeout_seconds), 0.5), 30.0)
        self.max_page_bytes = min(max(int(max_page_bytes), 1024), MAX_PAGE_BYTES)
        self.max_redirects = min(max(int(max_redirects), 0), MAX_REDIRECTS)
        self.resolve_host = resolve_host or _resolve_host

    async def read(self, url: str) -> dict[str, object]:
        current = _normalize_public_url(url)
        checked_hosts: set[str] = set()
        for redirect_count in range(self.max_redirects + 1):
            if current.host not in checked_hosts:
                await self._assert_public_host(current.host)
                checked_hosts.add(current.host)
                if not await self._robots_allow(current.url):
                    raise WebReadError("robots_disallowed", "The site's robots.txt does not allow this page to be fetched.")

            status, headers, body, response_url = await self._request(current.url, self.max_page_bytes)
            if status in {301, 302, 303, 307, 308}:
                location = headers.get("location")
                if not location:
                    raise WebReadError("redirect_without_location", "The page returned a redirect without a target.")
                if redirect_count >= self.max_redirects:
                    raise WebReadError("too_many_redirects", "The page exceeded the redirect limit.")
                current = _normalize_public_url(urljoin(response_url, location))
                continue
            if status < 200 or status >= 300:
                raise WebReadError("http_status_error", f"The public page returned HTTP {status}.")

            content_type = headers.get("content-type", "").split(";", 1)[0].strip().lower()
            is_pdf = content_type == "application/pdf" or body.startswith(b"%PDF-")
            if is_pdf:
                text = _extract_pdf(body)
                title = ""
                kind = "pdf"
            elif content_type in {"text/html", "application/xhtml+xml", ""}:
                parser = _VisibleTextParser()
                try:
                    parser.feed(body.decode(_charset(headers.get("content-type")), errors="replace"))
                except (LookupError, UnicodeError):
                    parser.feed(body.decode("utf-8", errors="replace"))
                text = " ".join(parser.parts)[:MAX_EXTRACTED_CHARS]
                title = " ".join(parser.title_parts)[:500]
                kind = "html"
            else:
                raise WebReadError("unsupported_content_type", "Only public HTML and PDF pages can be read.")

            if not text:
                raise WebReadError("empty_page", "No readable page text was found.")
            return {
                "url": response_url,
                "title": title,
                "content_type": content_type or ("application/pdf" if is_pdf else "text/html"),
                "document_type": kind,
                "text": text,
                "text_truncated": len(text) >= MAX_EXTRACTED_CHARS,
                "bytes_read": len(body),
                "status_code": status,
                "robots_allowed": True,
                "access": "public_anonymous",
            }
        raise WebReadError("too_many_redirects", "The page exceeded the redirect limit.")

    async def _robots_allow(self, page_url: str) -> bool:
        parsed = urlsplit(page_url)
        robots_url = urlunsplit((parsed.scheme, parsed.netloc, "/robots.txt", "", ""))
        try:
            status, _headers, body, _response_url = await self._request(robots_url, MAX_ROBOTS_BYTES)
        except WebReadError:
            # Fail closed when robots policy cannot safely be read.
            return False
        if status in {404, 410}:
            return True
        if status != 200:
            return False
        parser = RobotFileParser(robots_url)
        parser.parse(body.decode("utf-8", errors="replace").splitlines())
        return parser.can_fetch(USER_AGENT, page_url)

    async def _request(self, url: str, max_bytes: int) -> tuple[int, httpx.Headers, bytes, str]:
        normalized = _normalize_public_url(url)
        addresses = await self._assert_public_host(normalized.host)
        pinned_url = _replace_host(normalized.url, normalized.host, addresses[0])
        parsed = urlsplit(normalized.url)
        host_header = normalized.host
        if parsed.port is not None:
            host_header = f"{host_header}:{parsed.port}"
        headers = {
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/pdf;q=0.9,*/*;q=0.1",
            "Host": host_header,
        }
        # httpcore forwards this extension as `server_hostname` to AnyIO's TLS
        # stream, which expects a string and performs its own IDNA encoding.
        extensions = {"sni_hostname": normalized.host} if parsed.scheme == "https" else {}
        try:
            async with httpx.AsyncClient(
                timeout=self.timeout_seconds,
                transport=self.transport,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                request = client.build_request(
                    "GET", pinned_url, headers=headers, extensions=extensions
                )
                response = await client.send(request, stream=True, follow_redirects=False)
                try:
                    length = response.headers.get("content-length")
                    if length:
                        try:
                            if int(length) > max_bytes:
                                raise WebReadError("download_too_large", "The page is larger than the configured download limit.")
                        except ValueError:
                            pass
                    body = bytearray()
                    async for chunk in response.aiter_bytes():
                        body.extend(chunk)
                        if len(body) > max_bytes:
                            raise WebReadError("download_too_large", "The page is larger than the configured download limit.")
                    return response.status_code, response.headers, bytes(body), normalized.url
                finally:
                    await response.aclose()
        except WebReadError:
            raise
        except httpx.TimeoutException as exc:
            raise WebReadError("fetch_timeout", "The public page request timed out.") from exc
        except httpx.RequestError as exc:
            raise WebReadError("fetch_failed", "The public page could not be fetched.") from exc

    async def _assert_public_host(self, host: str) -> Sequence[str]:
        try:
            values = await asyncio.to_thread(self.resolve_host, host)
        except (OSError, socket.gaierror, ValueError) as exc:
            raise WebReadError("dns_resolution_failed", "The page host could not be resolved safely.") from exc
        if not values:
            raise WebReadError("dns_resolution_failed", "The page host did not resolve to a public address.")
        for value in values:
            try:
                address = ipaddress.ip_address(value)
            except ValueError as exc:
                raise WebReadError("dns_resolution_failed", "The page host returned an invalid network address.") from exc
            if not address.is_global:
                raise WebReadError("ssrf_address_blocked", "Private, local, or reserved network destinations are blocked.")
        return tuple(values)


class _NormalizedURL:
    def __init__(self, url: str, host: str) -> None:
        self.url = url
        self.host = host


def _normalize_public_url(url: str) -> _NormalizedURL:
    if not isinstance(url, str) or len(url) > 4096:
        raise WebReadError("invalid_url", "A valid public HTTP(S) URL is required.")
    try:
        parsed = urlsplit(url.strip())
        port = parsed.port
    except ValueError as exc:
        raise WebReadError("invalid_url", "The page URL is malformed.") from exc
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").rstrip(".").lower()
    if scheme not in {"http", "https"} or not host or parsed.username or parsed.password:
        raise WebReadError("invalid_url", "Only anonymous public HTTP(S) URLs are allowed.")
    if port not in {None, 80, 443}:
        raise WebReadError("port_blocked", "Only standard HTTP and HTTPS ports are allowed.")
    if host in {"localhost", "localhost.localdomain"} or host.endswith((".localhost", ".local", ".internal", ".lan", ".home", ".test", ".invalid")):
        raise WebReadError("ssrf_host_blocked", "Local and reserved hostnames are blocked.")
    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise WebReadError("invalid_url", "The page hostname is invalid.") from exc
    try:
        direct = ipaddress.ip_address(host)
    except ValueError:
        direct = None
    if direct is not None and not direct.is_global:
        raise WebReadError("ssrf_address_blocked", "Private, local, or reserved network destinations are blocked.")
    netloc = f"[{host}]" if ":" in host else host
    if port is not None:
        netloc = f"{netloc}:{port}"
    normalized = urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))
    return _NormalizedURL(normalized, host)


def _resolve_host(host: str) -> Sequence[str]:
    answers = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return tuple(dict.fromkeys(item[4][0] for item in answers))


def _replace_host(url: str, original_host: str, resolved_address: str) -> str:
    parsed = urlsplit(url)
    host = f"[{resolved_address}]" if ":" in resolved_address else resolved_address
    netloc = host + (f":{parsed.port}" if parsed.port is not None else "")
    return urlunsplit((parsed.scheme, netloc, parsed.path or "/", parsed.query, ""))


def _charset(content_type: str | None) -> str:
    if content_type:
        for part in content_type.split(";")[1:]:
            key, separator, value = part.strip().partition("=")
            if separator and key.strip().lower() == "charset":
                return value.strip().strip('"\'')
    return "utf-8"


def _extract_pdf(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise WebReadError("pdf_reader_unavailable", "PDF extraction is unavailable until the pypdf dependency is installed.") from exc
    try:
        reader = PdfReader(io.BytesIO(content), strict=False)
        text_parts = []
        for page in reader.pages[:MAX_PDF_PAGES]:
            text = page.extract_text() or ""
            if text:
                text_parts.append(text)
            if sum(map(len, text_parts)) >= MAX_EXTRACTED_CHARS:
                break
        return " ".join(" ".join(text_parts).split())[:MAX_EXTRACTED_CHARS]
    except Exception as exc:
        raise WebReadError("pdf_parse_failed", "The public PDF could not be parsed safely.") from exc
