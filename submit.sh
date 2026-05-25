#!/usr/bin/env bash
# AdaptFM competition submission script.
#
# Usage:
#   ./submit.sh [VARIANT]
#
# Variants (first argument):
#   mtp  — vLLM + MTP speculative decoding, FP16 weights  [DEFAULT]
#           Proven 1.33x speedup. Submit this while waiting for EXL3.
#   gpu  — ExLlamaV3 EXL3 4bpw (~2-3x expected speedup)
#           Requires: exl3-base:latest image + /data/models/qwen-weights-exl3-4bpw
#
# The script will:
#   1. Check that a local latency eval result exists and passes the speedup gate.
#   2. Build the Docker image for the chosen variant.
#   3. Save it as a gzip tarball (reuses existing tarball if image tag matches).
#   4. Register the team, request an upload URL, upload, and confirm submission.
#
# Resume an interrupted upload:
#   The multipart uploader saves progress to /tmp/afm_upload_etags.json.
#   Re-run the same command to resume from where it left off.

set -euo pipefail

TEAM_ID="AFM-6bhtn7up"
API_BASE="https://79x0as8g44.execute-api.us-east-1.amazonaws.com/prod"
API_KEY="qoXdZQNYbX1s4wnhmcBJG2APABlqjNVSao8CdM3j"

VARIANT="${1:-mtp}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Variant → Dockerfile + image tag ──────────────────────────────────────────
# NOTE: tarball MUST be named image.tar.gz — competition requirement.
# The Docker image tag is local-only; only the tarball content matters.
case "$VARIANT" in
  mtp)
    DOCKERFILE="Dockerfile.mtp"
    IMAGE_TAG="afm-submission:latest"
    MIN_SPEEDUP="1.15"
    ;;
  gpu)
    DOCKERFILE="Dockerfile.gpu"
    IMAGE_TAG="afm-submission:latest"
    MIN_SPEEDUP="1.50"   # EXL3 should give ≥1.5x; flag if not
    ;;
  *)
    echo "ERROR: Unknown variant '$VARIANT'. Choose: mtp | gpu"
    exit 1
    ;;
esac

# Tarball must be exactly image.tar.gz (competition API requirement).
TARBALL="image.tar.gz"
UPLOAD_RESP="/tmp/afm_upload_resp.json"

# If a tarball already exists from a different variant, remove it so we
# don't accidentally upload the wrong image.
LAST_VARIANT_FILE=".last_variant"
if [ -f "$TARBALL" ] && [ -f "$LAST_VARIANT_FILE" ]; then
  LAST_VARIANT=$(cat "$LAST_VARIANT_FILE")
  if [ "$LAST_VARIANT" != "$VARIANT" ]; then
    log "Removing stale tarball built for variant '$LAST_VARIANT' (now building '$VARIANT')"
    rm -f "$TARBALL"
  fi
fi

# ── Step 0: Latency gate ───────────────────────────────────────────────────────
EVAL_FILE="/tmp/local_eval_results.json"
if [ -f "$EVAL_FILE" ]; then
  AVG_MS=$(python3 -c "import json; d=json.load(open('$EVAL_FILE')); print(d.get('overall_avg_median_ms', 9999))")
  BASELINE_MS=4866
  SPEEDUP=$(python3 -c "print(f'{$BASELINE_MS / $AVG_MS:.2f}')")
  log "Latest latency eval: avg=${AVG_MS}ms  speedup=${SPEEDUP}x  (baseline=${BASELINE_MS}ms)"
  PASS=$(python3 -c "print('yes' if $BASELINE_MS / $AVG_MS >= $MIN_SPEEDUP else 'no')")
  if [ "$PASS" = "no" ]; then
    echo ""
    echo "⚠️  WARNING: Measured speedup ${SPEEDUP}x is below the ${MIN_SPEEDUP}x gate for variant '${VARIANT}'."
    echo "   Run a latency eval first: NUM_RUNS=20 EVAL_MODE=latency python3 run_eval_local.py"
    echo "   Continuing anyway (gate is advisory only)..."
    echo ""
  else
    log "✅ Latency gate passed: ${SPEEDUP}x ≥ ${MIN_SPEEDUP}x"
  fi
else
  log "⚠️  No latency eval results found at $EVAL_FILE — skipping gate check."
  log "   Run: NUM_RUNS=20 EVAL_MODE=latency python3 run_eval_local.py"
fi

# ── Step 1: Build ─────────────────────────────────────────────────────────────
# For the gpu variant, EXL3 weights live in /data/models/ (outside build context).
# Create a workspace symlink so Dockerfile.gpu's COPY picks them up.
EXL3_WEIGHTS_SRC="/data/models/qwen-weights-exl3-4bpw"
EXL3_WEIGHTS_LINK="qwen-weights-exl3-4bpw"
if [ "$VARIANT" = "gpu" ]; then
  if [ ! -d "$EXL3_WEIGHTS_SRC" ]; then
    echo "ERROR: EXL3 weights not found at $EXL3_WEIGHTS_SRC"
    echo "       Run quantisation first: python3 quantize_ex3.py"
    exit 1
  fi
  if [ ! -e "$EXL3_WEIGHTS_LINK" ]; then
    ln -s "$EXL3_WEIGHTS_SRC" "$EXL3_WEIGHTS_LINK"
    log "Symlinked EXL3 weights into build context ($EXL3_WEIGHTS_LINK → $EXL3_WEIGHTS_SRC)"
  fi
fi

log "Building Docker image '$IMAGE_TAG' from $DOCKERFILE ..."
DOCKER_BUILDKIT=1 docker build -f "$DOCKERFILE" -t "$IMAGE_TAG" .
log "Build complete."

# Clean up the temporary symlink after a successful build
if [ "$VARIANT" = "gpu" ] && [ -L "$EXL3_WEIGHTS_LINK" ]; then
  rm -f "$EXL3_WEIGHTS_LINK"
fi

# ── Step 2: Save ──────────────────────────────────────────────────────────────
if [ -f "$TARBALL" ]; then
  log "Reusing existing $TARBALL (delete it to force regeneration)"
else
  log "Saving image to $TARBALL ..."
  docker save "$IMAGE_TAG" | gzip > "$TARBALL"
  echo "$VARIANT" > "$LAST_VARIANT_FILE"
  log "Save complete."
fi
FILE_SIZE=$(stat -c%s "$TARBALL" 2>/dev/null || stat -f%z "$TARBALL")
log "Compressed size: $(python3 -c "print(f'{$FILE_SIZE/1e9:.2f} GB')")"

# ── Step 3: Register team (idempotent) ────────────────────────────────────────
log "Registering team $TEAM_ID ..."
curl -s -X POST "$API_BASE/register" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d "{\"team_id\": \"$TEAM_ID\"}" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print(' ', d)"

# ── Step 4: Request upload URL ────────────────────────────────────────────────
log "Requesting upload URL for $FILE_SIZE bytes ..."
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
  log "Uploading (single-part) ..."
  curl -X PUT "$UPLOAD_URL" \
    --upload-file "$TARBALL" \
    --progress-bar \
    -w "\nHTTP Status: %{http_code}\n"

# ── Step 5B: Multipart upload (> 5 GB, resumable) ────────────────────────────
else
  log "Uploading (multipart, resumable) ..."
  python3 - <<PYEOF
import json, os, subprocess, sys

API_BASE    = "https://79x0as8g44.execute-api.us-east-1.amazonaws.com/prod"
API_KEY     = "qoXdZQNYbX1s4wnhmcBJG2APABlqjNVSao8CdM3j"
UPLOAD_RESP = "$UPLOAD_RESP"
TARBALL     = "$TARBALL"
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

num_parts = resp["num_parts"]
part_size = resp["part_size"]
print(f"Uploading {num_parts} parts ({part_size//1024//1024} MB each) ...")

with open(TARBALL, "rb") as f:
    for part in resp["part_urls"]:
        n = part["part_number"]
        if n in done:
            f.seek(part_size, 1)
            continue
        chunk = f.read(part_size)
        if not chunk:
            break
        tmp = f"/tmp/afm_upload_part_{n}.bin"
        with open(tmp, "wb") as pf:
            pf.write(chunk)
        result = subprocess.run(
            ["curl", "-s", "-X", "PUT", part["upload_url"],
             "--upload-file", tmp, "-D", "-", "-o", "/dev/null"],
            capture_output=True, text=True,
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

print("All parts uploaded. Completing multipart upload ...")
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
    capture_output=True, text=True,
)
print(json.dumps(json.loads(result.stdout), indent=2))
if os.path.exists(ETAGS_FILE):
    os.remove(ETAGS_FILE)
PYEOF
fi

# ── Step 6: Confirm submission ────────────────────────────────────────────────
S3_KEY=$(python3 -c "import json; print(json.load(open('$UPLOAD_RESP'))['s3_key'])")
log "Confirming submission (s3_key=$S3_KEY) ..."
SUBMIT_RESP=$(curl -s -X POST "$API_BASE/submit" \
  -H "Content-Type: application/json" \
  -H "x-api-key: $API_KEY" \
  -d "{\"team_id\": \"$TEAM_ID\", \"s3_key\": \"$S3_KEY\"}")
echo "$SUBMIT_RESP"

SUBMISSION_ID=$(echo "$SUBMIT_RESP" | python3 -c \
  "import sys,json; d=json.load(sys.stdin); print(d.get('submission_id','unknown'))")
log "Submission complete!"
log "  Variant:       $VARIANT"
log "  Image:         $IMAGE_TAG"
log "  Submission ID: $SUBMISSION_ID"
log "  Evaluation:    90–100 minutes"
log "  Leaderboard:   https://d1krc5fcnf73gi.cloudfront.net"
