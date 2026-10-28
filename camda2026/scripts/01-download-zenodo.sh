#!/usr/bin/env bash
# Download CAMDA 2026 AMR pangenome data from Zenodo (record 19824655)
# into the shared project folder on Alnitak and verify MD5 checksums.
# Usage: bash 01-download-zenodo.sh [DEST_DIR]
set -euo pipefail

DEST="${1:-/files2/camda2026/data}"
BASE="https://zenodo.org/api/records/19824655/files"

declare -A MD5=(
  [pangenome.json]=7cb80479cf3c838b16b398d3e79523cb
  [pangenome.npz]=a5448d11350a9a7eb11b8fa87eb86487
  [pangenome.features.txt]=efeb1f79e421f56c264d4aeff6a520dd
  [pangenome.metadata.csv]=711c8dacb56c509fefabe283ec2a97d7
)

mkdir -p "$DEST"
cd "$DEST"

for f in "${!MD5[@]}"; do
  if [[ -f "$f" ]] && echo "${MD5[$f]}  $f" | md5sum -c --status; then
    echo "[skip] $f already present and verified"
    continue
  fi
  echo "[get ] $f"
  # Download to a .part file so an interrupted run never leaves a "valid-looking" file
  wget -q --show-progress -c -O "$f.part" "$BASE/$f/content"
  echo "${MD5[$f]}  $f.part" | md5sum -c --status || { echo "[FAIL] MD5 mismatch: $f" >&2; exit 1; }
  mv "$f.part" "$f"
  echo "[ ok ] $f"
done

# Read-only so nobody overwrites the shared raw data by accident
chmod a-w "$DEST"/pangenome.* 2>/dev/null || echo "[warn] could not chmod (files owned by another user)"
ls -lh "$DEST"
