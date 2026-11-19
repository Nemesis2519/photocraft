#!/bin/sh
# Remove Background on every photo in PHOTOS with each MODEL, through the release CLI and the app's
# model store; cut-outs go to OUT/<model>/<photo>.png, and each photo's time is printed.
# usage: run_models.sh PHOTOS OUT MODEL...
PHOTOS="$1"; OUT="$2"; shift 2
export PHOTOCRAFT_MODEL_DIR="${PHOTOCRAFT_MODEL_DIR:-$APPDATA/Photocraft/Models}"
CLI="${CLI:-$(dirname "$0")/../../target/agent-ml/release/photocraft-cli.exe}"
for model in "$@"; do
  mkdir -p "$OUT/$model"
  for f in "$PHOTOS"/*.jpg; do
    b=$(basename "$f" .jpg)
    start=$(date +%s%N)
    if "$CLI" run "$(cygpath -w "$f")" --cmd layer.removeBackground --params "{\"model\":\"$model\"}" \
        --out "$(cygpath -w "$OUT/$model/$b.png")" >/dev/null 2>"$OUT/$model/$b.err"; then
      echo "$model $b $(( ($(date +%s%N) - start) / 1000000 )) ms"
      rm -f "$OUT/$model/$b.err"
    else
      echo "$model $b FAILED: $(tail -1 "$OUT/$model/$b.err")"
    fi
  done
done
