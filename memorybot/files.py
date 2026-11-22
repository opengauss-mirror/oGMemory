"""Immutable Bot files on an independently configured AGFS mount."""

from __future__ import annotations

import base64
import codecs
import hashlib
import http.client
import ipaddress
import json
import os
import re
import socket
import ssl
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import quote, urljoin, urlsplit

import requests

MAX_FILE_BYTES = 50 * 1024 * 1024
MIME_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".csv": "text/csv",
}


class _PageText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.title = [], []
        self.hidden = 0
        self.in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden += 1
        if tag == "title":
            self.in_title = True
        if tag in {"p", "div", "br", "li", "h1", "h2", "h3", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg"}:
            self.hidden = max(0, self.hidden - 1)
        if tag == "title":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)
        elif not self.hidden:
            self.parts.append(data)


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, port, address, timeout):
        ca_bundle = os.environ.get("REQUESTS_CA_BUNDLE") or os.environ.get("CURL_CA_BUNDLE")
        context = ssl.create_default_context(
            cafile=ca_bundle if ca_bundle and not os.path.isdir(ca_bundle) else None,
            capath=ca_bundle if ca_bundle and os.path.isdir(ca_bundle) else None,
        )
        if not ca_bundle and not (
            os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR")
        ):
            context.load_verify_locations(cafile=requests.certs.where())
        super().__init__(host, port, timeout=timeout, context=context)
        self.address = address

    def connect(self):
        raw = socket.create_connection((self.address, self.port), self.timeout)
        self.sock = self._context.wrap_socket(raw, server_hostname=self.host)


def fetch_page(url: str, *, allowed_hosts=(), timeout=30, max_bytes=MAX_FILE_BYTES) -> dict:
    """Resolve and pin each public HTTP destination, including every redirect."""
    allowed_hosts = {host.lower() for host in allowed_hosts}
    for _ in range(6):
        if len(url) > 2048:
            raise ValueError("网页地址超过 2048 字符")
        parsed = urlsplit(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            raise ValueError("网页地址必须是公开 HTTP(S) 地址")
        host = parsed.hostname.encode("idna").decode("ascii").lower()
        if allowed_hosts and host not in allowed_hosts:
            raise ValueError("网页地址超出允许范围")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in {80, 443}:
            raise ValueError("网页端口必须为 80 或 443")
        addresses = {
            str(row[4][0]) for row in socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        }
        if not addresses or any(
            not ipaddress.ip_address(address).is_global for address in addresses
        ):
            raise ValueError("网页地址必须解析到公开网络")
        address = sorted(addresses)[0]
        connection = (
            _PinnedHTTPS(host, port, address, timeout)
            if parsed.scheme == "https"
            else http.client.HTTPConnection(address, port, timeout=timeout)
        )
        try:
            authority = f"[{host}]" if ":" in host else host
            if parsed.port:
                authority += f":{port}"
            target = parsed.path or "/"
            if parsed.query:
                target += "?" + parsed.query
            connection.request(
                "GET",
                quote(target, safe="/?:@!$&'()*+,;=%[]"),
                headers={
                    "Host": authority,
                    "Accept-Encoding": "identity",
                    "User-Agent": "oGMemory-memoryBot/1",
                },
            )
            response = connection.getresponse()
            if response.status in {301, 302, 303, 307, 308}:
                location = response.getheader("Location")
                if not location:
                    raise ValueError("网页重定向缺少目标地址")
                url = urljoin(url, location)
                continue
            if response.status != 200:
                raise ValueError(f"网页返回 HTTP {response.status}")
            if response.getheader("Content-Encoding", "identity").lower() != "identity":
                raise ValueError("网页须返回原始正文编码")
            mime = response.getheader("Content-Type", "text/html").split(";", 1)[0].lower()
            if mime not in {
                "text/html",
                "application/xhtml+xml",
                "text/plain",
                "text/markdown",
                "text/csv",
            }:
                raise ValueError("网页内容须为文本或 HTML")
            chunks, count = [], 0
            while chunk := response.read(min(65536, max_bytes - count + 1)):
                count += len(chunk)
                if count > max_bytes:
                    raise ValueError("网页超过 50MiB")
                chunks.append(chunk)
            data = b"".join(chunks)
            charset = response.headers.get_content_charset() or "utf-8-sig"
            try:
                encoding = codecs.lookup(charset).name
                text = data.decode("utf-8-sig" if encoding == "utf-8" else encoding)
            except LookupError:
                text = data.decode("utf-8-sig")
            title = host
            if mime in {"text/html", "application/xhtml+xml"}:
                parser = _PageText()
                parser.feed(text)
                title = "".join(parser.title).strip() or title
                text = "\n".join(
                    line.strip() for line in "".join(parser.parts).splitlines() if line.strip()
                )
            return {
                "data": data,
                "text": text,
                "name": title[:255],
                "mime_type": mime,
                "source_url": url,
            }
        finally:
            connection.close()
    raise ValueError("网页重定向次数超过限制")


def parse_text(data: bytes, name: str, mime_type: str | None = None) -> dict:
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("文件超过 50MiB")
    mime = MIME_TYPES.get(PurePosixPath(name).suffix.lower())
    if mime is None:
        raise ValueError("文件类型须为 TXT、Markdown 或 CSV")
    text = data.decode("utf-8-sig")
    if "\x00" in text:
        raise ValueError("文件内容须为 UTF-8 文本")
    return {
        "text": text,
        "mime_type": mime,
        "summary": text[:4096],
        "parsed_ranges": (
            [{"unit": "line", "start": 1, "end": len(text.splitlines())}] if text else []
        ),
    }


class AGFSFileStore:
    def __init__(
        self,
        base_url: str,
        mount_prefix: str,
        *,
        client=None,
        timeout=30,
        allowed_hosts=(),
    ):
        if client is None:
            from pyagfs import AGFSClient

            client = AGFSClient(api_base_url=base_url, timeout=timeout)
        self.client = client
        self.mount_prefix = "/" + mount_prefix.strip("/")
        self.allowed_hosts = tuple(allowed_hosts)

    def _path(self, account_id, storage_key):
        if not re.fullmatch(r"[A-Za-z0-9_.@-]{1,128}", account_id) or account_id in {
            ".",
            "..",
        }:
            raise ValueError("租户标识无效")
        parts = storage_key.split("/")
        if not storage_key or any(part in {"", ".", ".."} for part in parts) or "\\" in storage_key:
            raise ValueError("文件键无效")
        return f"{self.mount_prefix}/{account_id}/{storage_key}"

    def _parents(self, path):
        root = self.mount_prefix
        parts = path[len(root) :].strip("/").split("/")[:-1]
        for part in parts:
            root += "/" + part
            try:
                self.client.mkdir(root)
            except Exception as exc:
                if "exist" not in str(exc).lower():
                    raise

    def publish(
        self,
        account_id: str,
        file_id: str,
        data: bytes,
        *,
        name="content.md",
        mime_type=None,
    ) -> dict:
        if len(data) > MAX_FILE_BYTES:
            raise ValueError("文件超过 50MiB")
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", file_id):
            raise ValueError("文件标识无效")
        digest = hashlib.sha256(data).hexdigest()
        key = f"files/{file_id}/{digest}/content"
        path = self._path(account_id, key)
        self._parents(path)
        self.client.write(path, data)
        stored = self.client.read(path)
        if len(stored) != len(data) or hashlib.sha256(stored).hexdigest() != digest:
            raise OSError("共享文件校验未完成")
        return {
            "storage_key": key,
            "sha256": digest,
            "size_bytes": len(data),
            "name": name[:255],
            "mime_type": mime_type
            or MIME_TYPES.get(PurePosixPath(name).suffix.lower(), "text/plain"),
        }

    def read(self, account_id, storage_key, *, offset=0, size=-1):
        return self.client.read(self._path(account_id, storage_key), offset=offset, size=size)

    def write_parsed(self, account_id, storage_key, parsed):
        key = str(PurePosixPath(storage_key).with_name("parsed.json"))
        self.client.write(
            self._path(account_id, key),
            json.dumps(parsed, ensure_ascii=False).encode("utf-8"),
        )
        return key

    def fetch(self, url):
        return fetch_page(url, allowed_hosts=self.allowed_hosts)

    @staticmethod
    def _missing_object(exc):
        message = str(exc).lower()
        return any(marker in message for marker in ("not found", "404", "not exist"))

    def _entries(self, directory):
        try:
            return self.client.ls(directory)
        except Exception as exc:
            if not self._missing_object(exc):
                raise
            return []

    def delete_file_objects(self, account_id, file_id):
        """Remove revisions published before their storage key was saved."""
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", file_id):
            raise ValueError("文件标识无效")
        prefix = f"files/{file_id}"
        for row in self._entries(self._path(account_id, prefix)):
            digest = row["name"]
            if row["isDir"] and re.fullmatch(r"[a-f0-9]{64}", digest):
                self.delete(account_id, f"{prefix}/{digest}/content")

    def delete(self, account_id, storage_key):
        directory = str(PurePosixPath(self._path(account_id, storage_key)).parent)
        entries = self._entries(directory)
        if not entries:
            return
        if any(row["name"] == "backup.json" for row in entries):
            raise ValueError("文件仍受备份标记保护")
        for key in (
            storage_key,
            str(PurePosixPath(storage_key).with_name("parsed.json")),
        ):
            path = self._path(account_id, key)
            try:
                self.client.rm(path)
            except Exception as exc:
                if not self._missing_object(exc):
                    raise

    @staticmethod
    def _scan_path(relative):
        parts = relative.split("/") if relative else []
        patterns = [
            r"[A-Za-z0-9_.@-]{1,128}",
            r"files",
            r"[A-Za-z0-9_-]{1,64}",
            r"[a-f0-9]{64}",
        ]
        return len(parts) <= 4 and all(
            part not in {".", ".."} and re.fullmatch(patterns[index], part)
            for index, part in enumerate(parts)
        )

    def list_objects(self, limit=32, cursor=None):
        limit = max(1, min(100, int(limit)))
        stack = (
            json.loads(base64.urlsafe_b64decode(cursor)) if cursor else [{"path": "", "after": ""}]
        )
        if (
            not isinstance(stack, list)
            or len(stack) > 5
            or any(not self._scan_path(frame["path"]) for frame in stack)
        ):
            raise ValueError("文件扫描游标无效")
        items, cache, reads = [], {}, 0
        while stack and len(items) < limit and reads < limit * 8:
            frame = stack[-1]
            relative = frame["path"]
            if relative not in cache:
                cache[relative] = sorted(
                    self.client.ls(self.mount_prefix + ("/" + relative if relative else "")),
                    key=lambda row: row["name"],
                )
                reads += 1
            row = next((row for row in cache[relative] if row["name"] > frame["after"]), None)
            if row is None:
                stack.pop()
                continue
            frame["after"] = row["name"]
            candidate = relative + "/" + row["name"] if relative else row["name"]
            if row["isDir"]:
                if self._scan_path(candidate):
                    stack.append({"path": candidate, "after": ""})
                continue
            parts = relative.split("/")
            if len(parts) == 4 and row["name"] == "content":
                items.append(
                    {
                        "account_id": parts[0],
                        "storage_key": "/".join(parts[1:]) + "/content",
                        "file_id": parts[2],
                        "modified_at": row["modTime"],
                        "backup_protected": any(
                            entry["name"] == "backup.json" for entry in cache[relative]
                        ),
                    }
                )
        return {
            "items": items,
            "next_cursor": (
                base64.urlsafe_b64encode(json.dumps(stack).encode()).decode() if stack else None
            ),
        }
