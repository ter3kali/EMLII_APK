# EMLII for Android — build your own APK

You can't build an APK on the phone itself, so this project builds it in the cloud (free) or on a Linux/WSL PC.
**Honest note:** I couldn't run the Android build here, so the first build may need a small fix.
If it fails, send me the red error lines from the build log and I'll fix them.

## Easiest: GitHub builds it for you (no Linux needed)
1. Make a free account at github.com → **New repository** (name: `emlii-apk`, **Private**).
2. **Add file → Upload files** → drag in *everything* from this folder (including `.github`).
   - If `.github` doesn't upload: **Add file → Create new file**, name it `.github/workflows/build-apk.yml`,
     and paste the contents of `build-apk.yml`.
3. Open the **Actions** tab → **Build EMLII APK** → **Run workflow**. First build takes ~20–40 minutes.
4. When it turns green, open the run → **Artifacts → EMLII-apk** → download, unzip → `emlii-1.0-arm64-v8a-debug.apk`.
5. Send the APK to your phone (cable, Drive, Telegram…).

## Or: build on Linux / Windows WSL2
```bash
./build_apk.sh        # → bin/emlii-1.0-arm64-v8a-debug.apk
```

## Install on the Redmi A3 Pro (HyperOS)
1. Open the APK from your file manager. When asked, allow **Install unknown apps** for that app, then go back.
2. Play Protect may say "unsafe app" (it's just not from the Play Store): **More details → Install anyway**.
3. Open **EMLII**. It shows the web UI full-screen; enter a name and chat.

Updates: build again and install over the old one — your memory is kept (the included `debug.keystore` signs every build
the same way; keep the repository private).

## What works
- Chat, learning, predictions, corrections, the sidebar tools: `!trade`, `!waifu`, `!hash`, `!crack`, `!encrypt`, …
- Everything is stored inside the app (uninstalling it deletes your memory).
- **AI model (optional):** the app can't run Ollama itself. Run Ollama on a PC on your Wi-Fi, or `pkg install ollama` in
  Termux on the same phone, then in the chat type:
  `!ai url http://192.168.1.20:11434` (Termux: the default `http://127.0.0.1:11434` already works)
  `!ai model llama3.2:1b` · `!ai status` · `!ai test`

## What doesn't (yet)
- **Voice input:** Android's WebView has no speech recognition, so the 🎤 voice mode won't work inside the app.
  Typed chat and spoken replies are fine. (While the app is open you can try `http://127.0.0.1:5000` in Chrome — Chrome
  does have speech recognition.)
- `!download` (videos) isn't bundled — yt-dlp and ffmpeg are too heavy for the APK.
- The desktop Tkinter window isn't part of this; the APK is the web UI in a native shell.

## If something goes wrong
- **Blank page / ERR_CLEARTEXT_NOT_PERMITTED:** in `buildozer.spec` remove the `#` from the
  `android.extra_manifest_application_arguments` line, then rebuild.
- **App closes immediately:** the reason is saved in the app's `crash.log`. Easiest way to read it: connect the phone to a PC
  and run `adb logcat | grep -i python`, or tell me what you see.
- **Build fails on a package:** send me the last ~40 lines of the log.
