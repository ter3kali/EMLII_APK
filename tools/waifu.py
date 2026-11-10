"""
!waifu — anime pictures and reaction GIFs from the nekos.best API.

    !waifu                    one random "waifu" picture
    !waifu hug                a "hug" reaction GIF
    !waifu neko 5             five "neko" pictures
    !waifu gif | img | random a random GIF tag / image tag / any tag
    !waifu hug 3 save         fetch 3 and download them to your Downloads folder (waifu/)
    !waifu tags               list every tag

Standard library only (no `requests` needed).
"""
import difflib
import json
import os
import random
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

COMMAND = "waifu"
HELP = "Anime images/GIFs: !waifu [tag] [count] [save] · !waifu tags"

BASE_URL = "https://nekos.best/api/v2"
IMAGE_TAGS = ["waifu", "neko", "husbando", "kitsune"]
GIF_TAGS = [
    "hug", "kiss", "pat", "cry", "cuddle", "dance",
    "blush", "happy", "smile", "smug", "wave", "wink",
    "poke", "nom", "bite", "slap", "kick", "punch",
    "laugh", "bored", "sleep", "yawn", "think", "nod",
    "run", "pout", "feed", "tickle", "yeet", "shrug",
    "highfive", "handshake", "stare", "baka", "shoot",
]
ALL_TAGS = IMAGE_TAGS + GIF_TAGS
MAX_COUNT = 10                      # per command (the original CLI allowed 30)


def _save_dir():
    """Settings → Folders → Downloads (EMLII_DOWNLOADS_DIR) / waifu, else data/downloads/waifu."""
    root = os.environ.get("EMLII_DOWNLOADS_DIR")
    return Path(root) / "waifu" if root else Path(os.environ.get("EMLII_DATA_DIR") or "data") / "downloads" / "waifu"


_HEADERS = {"User-Agent": "EMLII-waifu/1.0"}


# ── API client (ported from mywaifu0.py) ─────────────────────────────────────
def _get_json(url):
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def _item(tag, result):
    return {
        "url": result.get("url"),
        "tag": tag,
        "is_gif": tag in GIF_TAGS,
        "anime_name": result.get("anime_name"),
        "artist_name": result.get("artist_name"),
    }


def fetch_one(tag="waifu"):
    result = (_get_json(f"{BASE_URL}/{tag}").get("results") or [{}])[0]
    if not result.get("url"):
        raise RuntimeError("the API returned no URL")
    return _item(tag, result)


def fetch_many(tag="waifu", count=5):
    """`count` API calls in parallel; duplicates dropped."""
    def one(_):
        try:
            return fetch_one(tag)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=5) as pool:
        found = [x for x in pool.map(one, range(max(1, min(count, MAX_COUNT)))) if x]
    seen, out = set(), []
    for x in found:
        if x["url"] not in seen:
            seen.add(x["url"]); out.append(x)
    if not out:
        raise RuntimeError("could not fetch any results")
    return out


def download_image(url, save_dir=None):
    save_dir = Path(save_dir or _save_dir()); save_dir.mkdir(parents=True, exist_ok=True)
    name = re.sub(r"[^\w.-]", "_", url.split("?")[0].rstrip("/").split("/")[-1]) or "image"
    path = save_dir / name
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=15) as r, open(path, "wb") as f:
        while True:
            chunk = r.read(8192)
            if not chunk:
                break
            f.write(chunk)
    return path


def random_tag(gif_only=False, image_only=False):
    return random.choice(GIF_TAGS if gif_only else IMAGE_TAGS if image_only else ALL_TAGS)


# ── EMLII entry point ────────────────────────────────────────────────────────
def _reply(text):
    return {"type": "reply", "text": text}


def _tag_list():
    return ("Image tags:\n  " + ", ".join(IMAGE_TAGS) +
            "\n\nGIF reaction tags:\n  " + ", ".join(GIF_TAGS))


def _describe(i, item, saved=None, numbered=True):
    head = f"[{i}] " if numbered else ""
    lines = [f"{head}{item['url']}"]
    meta = []
    if item["anime_name"]:  meta.append(f"anime: {item['anime_name']}")
    if item["artist_name"]: meta.append(f"artist: {item['artist_name']}")
    if meta:
        lines.append("    " + " · ".join(meta) if numbered else " · ".join(meta))
    if saved:
        lines.append(("    " if numbered else "") + f"saved → {saved}")
    return "\n".join(lines)


def run(arg, ctx):
    words = (arg or "").lower().split()
    if words and words[0] in ("tags", "list", "help", "?"):
        return _reply(_tag_list())

    save = any(w in ("save", "-s", "--save") for w in words)
    count = 1
    tag = None
    for w in words:
        if w in ("save", "-s", "--save"):
            continue
        if w.isdigit():
            count = int(w)
        elif w in ("gif", "gifs"):
            tag = random_tag(gif_only=True)
        elif w in ("img", "image", "images"):
            tag = random_tag(image_only=True)
        elif w == "random":
            tag = random_tag()
        elif tag is None:
            tag = w
    tag = tag or "waifu"
    count = max(1, min(count, MAX_COUNT))

    if tag not in ALL_TAGS:
        close = difflib.get_close_matches(tag, ALL_TAGS, n=3)
        hint = f"Did you mean: {', '.join(close)}?\n\n" if close else ""
        return _reply(f"Unknown tag '{tag}'.\n{hint}{_tag_list()}")

    try:
        items = fetch_many(tag, count) if count > 1 else [fetch_one(tag)]
    except urllib.error.HTTPError as e:
        return _reply(f"nekos.best returned an error: HTTP {e.code}.")
    except urllib.error.URLError as e:
        return _reply(f"Couldn't reach nekos.best ({e.reason}). Check your internet connection.")
    except Exception as e:
        return _reply(f"Couldn't fetch '{tag}': {e}")

    kind = "GIF" if tag in GIF_TAGS else "image"
    lines = [f"{tag} · {kind}" + (f" × {len(items)}" if len(items) > 1 else "")]
    for i, item in enumerate(items, 1):
        saved = None
        if save:
            try:
                saved = download_image(item["url"])
            except Exception as e:
                saved = f"(download failed: {e})"
        lines.append(_describe(i, item, saved, numbered=len(items) > 1))
    out = _reply("\n".join(lines))
    out["images"] = [i["url"] for i in items]       # the desktop app shows these inline
    return out
