"""
!ai — choose which Ollama server and model EMLII's AI fallback uses.

    !ai                      status: server, model, and whether it answers
    !ai url <address>        e.g.  !ai url http://192.168.1.20:11434   (a PC, or Termux on this phone)
    !ai url default          back to this device (http://127.0.0.1:11434)
    !ai model <name>         e.g.  !ai model llama3.2:1b
    !ai test                 ask the model a quick question

The settings are saved in the data folder, so they stick after a restart.
Handy on Android, where you can't edit files.
"""
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

COMMAND = "ai"
HELP = "AI server/model: !ai status | url <addr> | model <name> | test"


def _core():
    return sys.modules.get("EMLII") or sys.modules.get("__main__")


def _data():
    return Path(os.environ.get("EMLII_DATA_DIR") or "data")


def _save(name, value):
    d = _data(); d.mkdir(parents=True, exist_ok=True)
    f = d / name
    if value:
        f.write_text(value)
    elif f.exists():
        f.unlink()


def _reply(text):
    return {"type": "reply", "text": text}


def _base(core):
    return core.OLLAMA_URL.rsplit("/api/generate", 1)[0]


def _models(core):
    with urllib.request.urlopen(_base(core) + "/api/tags", timeout=4) as r:
        return [m.get("name", "?") for m in json.loads(r.read().decode()).get("models", [])]


def _status(core):
    model = core.ACTIVE_MODEL or core.OLLAMA_MODEL
    lines = [f"AI server : {_base(core)}", f"AI model  : {model}"]
    try:
        names = _models(core)
        lines.append("Reachable : yes")
        lines.append("Installed : " + (", ".join(names[:6]) if names else "(none — run: ollama pull <model>)"))
        if names and not any(model == n or model.split(":")[0] in n for n in names):
            lines.append(f"Heads-up  : '{model}' isn't installed there — use  !ai model <name>")
    except Exception as e:
        lines.append(f"Reachable : no ({getattr(e, 'reason', e)})")
        lines.append("Start Ollama there, or point to another one with  !ai url <address>")
    return "\n".join(lines)


def run(arg, ctx):
    core = _core()
    if core is None or not hasattr(core, "OLLAMA_URL"):
        return _reply("Couldn't reach EMLII's AI settings.")
    parts = (arg or "").split()
    sub = parts[0].lower() if parts else "status"

    if sub in ("status", "s", "info"):
        return _reply(_status(core))

    if sub == "url":
        if len(parts) < 2:
            return _reply("Usage: !ai url <address>   e.g. !ai url http://192.168.1.20:11434")
        val = parts[1].strip()
        if val.lower() in ("default", "reset", "local"):
            _save("ollama_url.txt", "")
            os.environ.pop("EMLII_OLLAMA_URL", None)
            core.OLLAMA_URL = core._ollama_url()
            return _reply("Using this device's Ollama again.\n\n" + _status(core))
        if not val.startswith(("http://", "https://")):
            val = "http://" + val
        _save("ollama_url.txt", val.rstrip("/"))
        os.environ.pop("EMLII_OLLAMA_URL", None)             # the saved choice now applies
        core.OLLAMA_URL = core._ollama_url()
        return _reply("Saved.\n\n" + _status(core))

    if sub == "model":
        if len(parts) < 2:
            return _reply("Usage: !ai model <name>   e.g. !ai model llama3.2:1b")
        name = " ".join(parts[1:])
        _save("ollama_model.txt", "" if name.lower() == "default" else name)
        core.OLLAMA_MODEL = core._pref("ollama_model.txt") or "hf.co/bartowski/SmolLM2-360M-Instruct-GGUF:Q5_K_M"
        return _reply("Saved.\n\n" + _status(core))

    if sub == "test":
        t0 = time.time()
        try:
            out = core.query_ollama("Say hello in five words.") if hasattr(core, "query_ollama") else None
        except Exception as e:
            out = None
            err = str(e)
        else:
            err = ""
        if out:
            return _reply(f"{out}\n\n({time.time() - t0:.1f}s)")
        return _reply("No answer from the AI." + (f" ({err})" if err else "") + "\n\n" + _status(core))

    return _reply("Options: !ai status | !ai url <address> | !ai model <name> | !ai test")
