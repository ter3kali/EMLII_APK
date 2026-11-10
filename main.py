"""
EMLII for Android.

python-for-android's "webview" bootstrap runs this file and shows
http://127.0.0.1:5000 full-screen in a WebView. All we have to do is start EMLII's
own web server (Flask) on that port, with its folders in the app's private storage.
"""
import os
import shutil
import sys
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PORT = 5000


def private_dir():
    try:
        from android.storage import app_storage_path     # only exists on Android
        return Path(app_storage_path())
    except Exception:
        return Path.home() / ".emlii-android"            # running on a PC for testing


HOME = private_dir()
HOME.mkdir(parents=True, exist_ok=True)
os.chdir(HOME)

# EMLII keeps everything under these folders (see EMLII.py: EMLII_*_DIR)
os.environ.setdefault("EMLII_DATA_DIR", str(HOME / "data"))
os.environ.setdefault("EMLII_TOOLS_DIR", str(HOME / "tools"))
os.environ.setdefault("EMLII_MODELS_DIR", str(HOME / "models"))
os.environ.setdefault("EMLII_DOWNLOADS_DIR", str(HOME / "downloads"))
os.environ["EMLII_WEB_HOST"] = "127.0.0.1"               # only this app can reach the server
os.environ.setdefault("EMLII_OLLAMA_TIMEOUT", "90")

# Optional AI: in the chat, type  !ai url http://<address>:11434  (a PC on your Wi-Fi, or Termux's Ollama on this
# phone) and  !ai model <name>. The choice is saved in the data folder above.

# plugins: refresh the bundled ones each launch; any extra .py you drop in tools/ stays
(HOME / "tools").mkdir(parents=True, exist_ok=True)
for f in (APP_DIR / "tools").glob("*.py"):
    shutil.copy2(f, HOME / "tools" / f.name)

sys.path.insert(0, str(APP_DIR))
try:
    import EMLII as core
    core.load_tools()
    core.run_web(PORT)                                   # blocks; the WebView shows it
except Exception:
    (HOME / "crash.log").write_text(traceback.format_exc())
    raise
