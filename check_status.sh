#!/usr/bin/env bash
# =============================================================
#  check_status.sh
#  Run as the "dt" user to check the DTown Bot service status.
#
#  Usage:
#    chmod +x check_status.sh
#    ./check_status.sh
# =============================================================

SERVICE_NAME="dtbot.service"

echo ""
echo "============================================================"
echo "  DTown Bot — Service Status"
echo "============================================================"

echo ""
echo "[1] Current status:"
systemctl --user status "$SERVICE_NAME" --no-pager

echo ""
echo "[2] Last 50 log lines:"
journalctl --user -u "$SERVICE_NAME" -n 50 --no-pager

echo ""
echo "============================================================"
echo "  Useful commands:"
echo "    journalctl --user -u dtbot.service -f      # live logs"
echo "    systemctl --user restart dtbot.service      # restart"
echo "    systemctl --user stop dtbot.service         # stop"
echo "    systemctl --user start dtbot.service        # start"
echo "============================================================"
