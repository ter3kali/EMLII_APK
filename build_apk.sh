#!/usr/bin/env bash
# Build the APK on Linux or Windows WSL2 (Ubuntu 22.04 recommended). Takes 20–40 min the first time.
set -euo pipefail
cd "$(dirname "$0")"
sudo apt-get update
sudo apt-get install -y git zip unzip openjdk-17-jdk python3-pip python3-venv autoconf automake libtool \
  pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev libtinfo5 cmake libffi-dev libssl-dev libltdl-dev build-essential
python3 -m venv .venv && . .venv/bin/activate
pip install --upgrade pip buildozer "cython<3" virtualenv
mkdir -p ~/.android
[ -f debug.keystore ] && cp debug.keystore ~/.android/debug.keystore   # same signing key every build
buildozer -v android debug
echo; echo "APK: $(ls bin/*.apk)"
