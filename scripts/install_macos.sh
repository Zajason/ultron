#!/bin/bash
# Run Ultron in the background on macOS, as a launchd user agent (starts at login, restarts on crash).
#
#   scripts/install_macos.sh install     install and start
#   scripts/install_macos.sh uninstall   stop and remove
#   scripts/install_macos.sh status      is it running?
#   scripts/install_macos.sh logs        follow the log
#
# Needs: uv, a trained wake word (scripts/train_wakeword.py), and API keys in .env.
# Microphone: macOS asks for microphone access the first time the agent records. If the log says
# the mic is sending silence, allow the listed app under System Settings > Privacy & Security >
# Microphone, then: scripts/install_macos.sh uninstall && scripts/install_macos.sh install
set -euo pipefail

LABEL="ai.ultron.assistant"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
LOGDIR="$HOME/.ultron/logs"
UV="$(command -v uv || true)"
DOMAIN="gui/$(id -u)"

install() {
  [[ -n "$UV" ]] || { echo "uv not found (brew install uv)"; exit 1; }
  [[ -f "$ROOT/.env" ]] || { echo "missing $ROOT/.env (see .env.example)"; exit 1; }
  [[ -f "$ROOT/src/ultron/models/ultron_wakeword.onnx" ]] || { echo "no wake-word model: uv run --group train python scripts/train_wakeword.py all"; exit 1; }
  (cd "$ROOT" && "$UV" sync --quiet)
  mkdir -p "$LOGDIR" "$(dirname "$PLIST")"
  cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$ROOT/.venv/bin/ultron</string>
    <string>--config-dir</string><string>$ROOT</string>
    <string>live</string><string>--wake</string><string>--daemon</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><dict><key>SuccessfulExit</key><false/></dict>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$LOGDIR/ultron.log</string>
  <key>StandardErrorPath</key><string>$LOGDIR/ultron.log</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
</dict>
</plist>
EOF
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$PLIST"
  echo "installed: $PLIST"
  echo "logs: $LOGDIR/ultron.log"
}

uninstall() {
  launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
  rm -f "$PLIST"
  echo "removed $LABEL"
}

case "${1:-}" in
  install) install ;;
  uninstall) uninstall ;;
  status) launchctl print "$DOMAIN/$LABEL" 2>/dev/null | grep -E "state|pid|last exit" || echo "not installed" ;;
  logs) tail -f "$LOGDIR/ultron.log" ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
