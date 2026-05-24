#!/usr/bin/env bash
# AdaptFM competition submission script.
# Usage: ./submit.sh [IMAGE_TAG]
# Default image tag: afm-gpu:latest

set -euo pipefail

TEAM_ID="AFM-6bhtn7up"
API_BASE="https://79x0as8g44.execute-api.us-east-1.amazonaws.com/prod"
API_KEY="qoXdZQNYbX1s4wnhmcBJG2APABlqjNVSao8CdM3j"
IMAGE_TAG="${1:-afm-gpu:latest}"
TARBALL="image.tar.gz"
UPLOAD_RESP="/tmp/afm_upload_resp.json"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Step 1: Build ─────────────────────────────────────────────────────────────
log "Building Docker image: $IMAGE_TAG"
docker build -f Dockerfile.gpu -t "$IMAGE_TAG" .

# ── Step 2: Save ──────────────────────────────────────────────────────────────
if [ -f "$TARBALL" ]; then
  log "Reusing existing $TARBALL (delete it to force regeneration)"
else
  log "Saving image to $TARBALL (this takes a few minutes)..."
  docker save "$IMAGE_TAG" | gzip > "$TARBALL"
fi
FILE_SIZE=$(stat -c%s "$TARBALL" 2>/dev/null || stat -f%z "$TARBALL")
log "Compressed size: $(python3 -c "print(f'{$FILE_SIZE/1e9:.2f} GB')")"

# ── Step 3: Register team (idempotent) ────────────────────────────────────────
log "Registering team $TEAM_ID..."
curl -s -X POST "$API_BASE/register" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d "{\"team_id\": \"$TEAM_ID\"}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(' ', d)"

# ── Step 4: Request upload URL ────────────────────────────────────────────────
log "Requesting upload URL for $FILE_SIZE bytes..."
curl -s -X POST "$API_BASE/upload-url" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d "{\"team_id\": \"$TEAM_ID\", \"file_size_bytes\": $FILE_SIZE}" \
  -o "$UPLOAD_RESP"

UPLOAD_TYPE=$(python3 -c "import json; print(json.load(open('$UPLOAD_RESP'))['upload_type'])")
log "Upload type: $UPLOAD_TYPE"

# ── Step 5A: Single-part upload (≤ 5 GB) ─────────────────────────────────────
if [ "$UPLOAD_TYPE" = "single" ]; then
  UPLOAD_URL=$(python3 -c "import json; print(json.load(open('$UPLOAD_RESP'))['upload_url'])")
  log "Uploading (single-part)..."
  curl -X PUT "$UPLOAD_URL" \
    --upload-file "$TARBALL" \
    --progress-bar \
    -w "\nHTTP Status: %{http_code}\n"

# ── Step 5B: Multipart upload (> 5 GB, resumable) ────────────────────────────
else
  log "Uploading (multipart)..."
  python3 - <<'PYEOF'
import json, os, subprocess, sys

API_BASE    = "https://79x0as8g44.execute-api.us-east-1.amazonaws.com/prod"
API_KEY     = "qoXdZQNYbX1s4wnhmcBJG2APABlqjNVSao8CdM3j"
UPLOAD_RESP = "/tmp/afm_upload_resp.json"
TARBALL     = "image.tar.gz"
ETAGS_FILE  = "/tmp/afm_upload_etags.json"

resp    = json.load(open(UPLOAD_RESP))
team_id = resp["s3_key"].split("/")[1]

# Resume support — skip already-uploaded parts
if os.path.exists(ETAGS_FILE):
    etags = json.load(open(ETAGS_FILE))
    done  = {e["part_number"] for e in etags}
    print(f"Resuming — {len(done)} parts already done: {sorted(done)}")
else:
    etags = []
    done  = set()

num_parts  = resp["num_parts"]
part_size  = resp["part_size"]
print(f"Uploading {num_parts} parts ({part_size//1024//1024} MB each)...")

with open(TARBALL, "rb") as f:
    for part in resp["part_urls"]:
        n = part["part_number"]
        if n in done:
            f.seek(part_size, 1)
            continue
        chunk = f.read(part_size)
        if not chunk:
            break
        tmp = f"/tmp/part_{n}.bin"
        with open(tmp, "wb") as pf:
            pf.write(chunk)
        result = subprocess.run(
            ["curl", "-s", "-X", "PUT", part["upload_url"],
             "--upload-file", tmp, "-D", "-", "-o", "/dev/null"],
            capture_output=True, text=True
        )
        os.remove(tmp)
        etag = ""
        for line in result.stdout.splitlines():
            if line.lower().startswith("etag:"):
                etag = line.split(":", 1)[1].strip().strip("\r").strip('"')
                break
        if not etag:
            print(f"  ERROR: No ETag for part {n}. Re-run to resume.")
            sys.exit(1)
        etags.append({"part_number": n, "etag": etag})
        done.add(n)
        with open(ETAGS_FILE, "w") as ef:
            json.dump(etags, ef)
        print(f"  Part {n}/{num_parts} done (etag={etag})")

print("All parts uploaded. Completing...")
body = json.dumps({
    "team_id":   team_id,
    "s3_key":    resp["s3_key"],
    "upload_id": resp["upload_id"],
    "parts":     etags,
})
result = subprocess.run(
    ["curl", "-s", "-X", "POST", f"{API_BASE}/complete-upload",
     "-H", "Content-Type: application/json",
     "-H", f"x-api-key: {API_KEY}",
     "-d", body],
    capture_output=True, text=True
)
print(json.dumps(json.loads(result.stdout), indent=2))
if os.path.exists(ETAGS_FILE):
    os.remove(ETAGS_FILE)
PYEOF
fi

# ── Step 6: Confirm submission ────────────────────────────────────────────────
S3_KEY=$(python3 -c "import json; print(json.load(open('$UPLOAD_RESP'))['s3_key'])")
log "Confirming submission (s3_key=$S3_KEY)..."
SUBMIT_RESP=$(curl -s -X POST "$API_BASE/submit" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d "{\"team_id\": \"$TEAM_ID\", \"s3_key\": \"$S3_KEY\"}")
echo "$SUBMIT_RESP"

SUBMISSION_ID=$(echo "$SUBMIT_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('submission_id','unknown'))")
log "Submission complete. ID: $SUBMISSION_ID"
log "Evaluation takes 90–100 minutes. Leaderboard: https://d1krc5fcnf73gi.cloudfront.net"
