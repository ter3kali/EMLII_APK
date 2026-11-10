[app]
title = EMLII
package.name = emlii
package.domain = org.emlii
source.dir = .
source.include_exts = py,png,json,txt,html
source.exclude_dirs = .github,bin,.buildozer,__pycache__
version = 1.0

# Flask and what it needs. (No Kivy: the "webview" bootstrap shows EMLII's web UI.)
requirements = python3,flask,werkzeug,jinja2,markupsafe,itsdangerous,click,blinker,android

orientation = portrait
fullscreen = 0
icon.filename = %(source.dir)s/icon.png
presplash.filename = %(source.dir)s/presplash.png
android.presplash_color = #0B0A10

# INTERNET: the web UI talks to the server inside this app; also !waifu, and an optional Ollama.
android.permissions = INTERNET,ACCESS_NETWORK_STATE

android.api = 33
android.minapi = 24
android.ndk = 25b
# 64-bit only (every recent phone, including the Redmi A3 Pro). Add armeabi-v7a for very old 32-bit phones.
android.archs = arm64-v8a
android.accept_sdk_license = True
android.release_artifact = apk

p4a.bootstrap = webview
p4a.port = 5000

# If the app opens to a blank page / ERR_CLEARTEXT_NOT_PERMITTED, uncomment this line
# (it allows http://127.0.0.1) and rebuild:
#android.extra_manifest_application_arguments = ./manifest_app_args.xml

[buildozer]
log_level = 2
warn_on_root = 1
