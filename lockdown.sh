#!/usr/bin/env bash
# =============================================================
#  lockdown.sh
#
#  Hardens the Raspberry Pi for production deployment.
#  Run this ONLY after setup_service.sh and a reboot test
#  confirm the chatbot starts correctly on its own.
#
#  What this script does:
#   [A] Boot → CLI autologin (no desktop, saves ~300MB RAM)
#   [B] SSH left as-is (password login kept)
#   [C] Sets a strong dt password you choose
#   [D] Keeps sudo on dt (needed for maintenance)
#   [E] Disables Bluetooth (not needed, reduces attack surface)
#   [F] Disables unnecessary services (cups, avahi, triggerhappy, hciuart)
#        NOTE: wpa_supplicant is NOT disabled — WiFi must stay on
#   [G] Makes the .env file readable only by "dt" user
#   [H] Locks the project directory permissions
#   [I] Enables PulseAudio as user service (required for audio at boot)
#
#  Usage:
#    chmod +x lockdown.sh
#    sudo ./lockdown.sh
# =============================================================

set -euo pipefail

# Must run as root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Run this script with sudo: sudo ./lockdown.sh"
    exit 1
fi

PROJECT_DIR="/home/dt/Desktop/DTown"
ADMIN_USER="dt"

echo ""
echo "============================================================"
echo "  DTown Bot — Production Lockdown"
echo "============================================================"
echo ""
echo "  ⚠️  WARNING: This will restrict system access."
echo "  Make sure the chatbot service is working before continuing."
echo ""
read -p "  Type YES to continue: " confirm
[ "$confirm" = "YES" ] || { echo "Aborted."; exit 0; }

# ── [A] Boot to CLI with autologin ────────────────────────
# REQUIRED: autologin opens the user session that PulseAudio
# attaches to. Without it, audio will not work at boot even
# though the dtbot process starts via linger.
echo ""
echo "[A] Configuring boot to CLI with autologin..."

systemctl set-default multi-user.target

mkdir -p /etc/systemd/system/getty@tty1.service.d/
cat > /etc/systemd/system/getty@tty1.service.d/autologin.conf << UNIT
[Service]
ExecStart=
ExecStart=-/sbin/agetty --autologin $ADMIN_USER --noclear %I \$TERM
UNIT

echo "  ✅ Boot target: CLI + autologin as '$ADMIN_USER'"
echo "     (Desktop GUI disabled — saves ~300 MB RAM for the chatbot)"

# ── [B] SSH — left as-is with password login ──────────────
echo ""
echo "[B] SSH — leaving as-is (password login kept)"
echo "  ✅ SSH unchanged — you can still connect with password over LAN"
echo "  ⚠️  Tip: make sure your router does not expose port 22 to the internet"

# ── [C] Set a strong dt password ──────────────────────────
echo ""
echo "[C] Setting password for user '$ADMIN_USER'..."
echo "  Choose a strong password."
passwd "$ADMIN_USER"
echo "  ✅ Password updated"

# ── [D] Sudo kept on dt ───────────────────────────────────
echo ""
echo "[D] Sudo access — keeping sudo on '$ADMIN_USER'"
echo "  ✅ Sudo unchanged — needed for reboot, apt, and maintenance over SSH"

# ── [E] Disable Bluetooth ─────────────────────────────────
echo ""
echo "[E] Disabling Bluetooth (not needed for chatbot)..."
systemctl disable bluetooth 2>/dev/null || true
systemctl stop    bluetooth 2>/dev/null || true

BOOT_CFG="/boot/config.txt"
[ -f "/boot/firmware/config.txt" ] && BOOT_CFG="/boot/firmware/config.txt"

if ! grep -q "dtoverlay=disable-bt" "$BOOT_CFG" 2>/dev/null; then
    echo "dtoverlay=disable-bt" >> "$BOOT_CFG"
fi
echo "  ✅ Bluetooth disabled"

# ── [F] Disable unnecessary services ──────────────────────
# ⚠️  wpa_supplicant is intentionally NOT in this list.
#     Disabling it kills WiFi, which kills Groq API calls.
echo ""
echo "[F] Disabling unnecessary background services..."

SERVICES_TO_DISABLE=(
    "cups"
    "cups-browsed"
    "avahi-daemon"
    "triggerhappy"
    "hciuart"
)

for svc in "${SERVICES_TO_DISABLE[@]}"; do
    if systemctl list-unit-files --quiet "$svc.service" &>/dev/null; then
        systemctl disable "$svc" 2>/dev/null || true
        systemctl stop    "$svc" 2>/dev/null || true
        echo "  ✅ $svc disabled"
    else
        echo "  ⏭  $svc not installed — skipping"
    fi
done

echo "  ✅ wpa_supplicant intentionally kept — WiFi must stay on for Groq API"

# ── [G] Lock down .env file ───────────────────────────────
echo ""
echo "[G] Securing .env file (contains API key)..."
if [ -f "$PROJECT_DIR/.env" ]; then
    chown "$ADMIN_USER:$ADMIN_USER" "$PROJECT_DIR/.env"
    chmod 600 "$PROJECT_DIR/.env"
    echo "  ✅ .env: permissions set to 600 (owner-only read/write)"
else
    echo "  ⚠️  .env not found at $PROJECT_DIR — skipping"
    echo "     Make sure it exists before running the chatbot"
fi

# ── [H] Lock project directory ────────────────────────────
echo ""
echo "[H] Locking project directory permissions..."
chown -R "$ADMIN_USER:$ADMIN_USER" "$PROJECT_DIR"
chmod 750 "$PROJECT_DIR"
echo "  ✅ $PROJECT_DIR: permissions set to 750"

# ── [I] Enable PulseAudio as user service ─────────────────
# This is critical. Without PulseAudio running at boot as a
# user service, dtbot.service will start but produce no audio.
echo ""
echo "[I] Enabling PulseAudio as a user service for '$ADMIN_USER'..."

sudo -u "$ADMIN_USER" XDG_RUNTIME_DIR="/run/user/$(id -u $ADMIN_USER)" \
    systemctl --user enable pulseaudio 2>/dev/null || true

sudo -u "$ADMIN_USER" XDG_RUNTIME_DIR="/run/user/$(id -u $ADMIN_USER)" \
    systemctl --user enable pulseaudio.socket 2>/dev/null || true

echo "  ✅ PulseAudio user service enabled"
echo "     It will start automatically when '$ADMIN_USER' session opens at boot"

# ── Summary ───────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Lockdown complete. Summary:"
echo ""
echo "  [A] Boot → CLI autologin as '$ADMIN_USER' (no desktop)"
echo "      └─ Required for PulseAudio + audio at boot"
echo "  [B] SSH: password login kept (unchanged)"
echo "  [C] dt password: updated"
echo "  [D] Sudo: kept on '$ADMIN_USER'"
echo "  [E] Bluetooth: disabled"
echo "  [F] Unnecessary services disabled (WiFi kept)"
echo "  [G] .env: chmod 600 (API key secured)"
echo "  [H] Project dir: chmod 750"
echo "  [I] PulseAudio: enabled as user service"
echo ""
echo "  Boot sequence after reboot:"
echo "    Power on → autologin as dt → PulseAudio starts"
echo "    → dtbot.service starts → chatbot speaks ✅"
echo ""
echo "  ⚠️  REBOOT NOW to verify everything works:"
echo "    sudo reboot"
echo ""
echo "  After reboot, check from another machine via SSH:"
echo "    journalctl --user -u dtbot.service -n 30"
echo "============================================================"
