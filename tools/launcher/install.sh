#!/bin/bash
# Installs pacbot-train and pacbot-capacity for everyone in the pacbot group.
# Run with sudo from this directory: sudo bash install.sh   (run it again after updating them)
set -euo pipefail
cd "$(dirname "$0")"
OWNER=${PACBOT_TRAIN_OWNER:-nassa}
STATE=/var/lib/pacbot-train
getent group pacbot >/dev/null || groupadd pacbot
install -m 755 pacbot-train pacbot-capacity /usr/local/bin/
# setgid: files belong to the group; sticky: only a file's owner can replace or remove it
install -d -m 3775 -o root -g pacbot "$STATE" "$STATE/jobs"
for f in lock seq; do
  [ -e "$STATE/$f" ] || install -m 664 -o root -g pacbot /dev/null "$STATE/$f"
done
[ -e "$STATE/capacity" ] || { echo 3 > "$STATE/capacity"; }
chown "$OWNER:pacbot" "$STATE/capacity"; chmod 644 "$STATE/capacity"
echo "Installed. State in $STATE:"
ls -la "$STATE"
