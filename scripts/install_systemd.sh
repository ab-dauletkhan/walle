#!/usr/bin/env bash
# Install (but do NOT enable) the WALL-E systemd unit so it can be
# triggered manually for testing and flipped to auto-start at boot
# only when the operator is confident the stack runs cleanly.
#
# Usage:
#   sudo bash scripts/install_systemd.sh            # copy + reload daemon
#   sudo bash scripts/install_systemd.sh --enable   # also enable at boot
#
# After install, control the robot with:
#   sudo systemctl start walle        # one-shot launch
#   sudo systemctl stop walle         # clean shutdown
#   sudo systemctl status walle       # running state
#   journalctl -u walle -f            # live log stream
#   sudo systemctl enable walle       # turn on auto-boot when ready
#   sudo systemctl disable walle      # turn auto-boot off

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC="$SCRIPT_DIR/walle.service"
DST="/etc/systemd/system/walle.service"

if [[ "$(id -u)" != "0" ]]; then
    echo "This script needs root to write /etc/systemd/system/. Re-run with sudo." >&2
    exit 1
fi

if [[ ! -f "$SRC" ]]; then
    echo "ERROR: $SRC not found" >&2
    exit 1
fi

echo "Installing $SRC → $DST"
cp "$SRC" "$DST"
chmod 644 "$DST"

echo "Reloading systemd..."
systemctl daemon-reload

if [[ "${1:-}" == "--enable" ]]; then
    echo "Enabling walle.service (will start at next boot)..."
    systemctl enable walle.service
    echo
    echo "✓ Auto-boot enabled. Start now with:  sudo systemctl start walle"
else
    echo
    echo "✓ Unit installed. Auto-boot is OFF."
    echo
    echo "Manual control:"
    echo "  sudo systemctl start walle    # launch"
    echo "  sudo systemctl stop walle     # shutdown"
    echo "  sudo systemctl status walle"
    echo "  journalctl -u walle -f        # follow logs"
    echo
    echo "When ready for boot-time auto-start:"
    echo "  sudo systemctl enable walle"
fi
