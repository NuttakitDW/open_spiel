#!/bin/sh
set -eu
repo_dir=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
model="$repo_dir/build-arena/model.bin"
if [ ! -s "$model" ]; then
  echo 'missing exported build-arena/model.bin; build and run arena_model_export first' >&2
  exit 1
fi
mkdir -p "$repo_dir/build-arena/dist"
cat > "$repo_dir/build-arena/arena_model.S" <<'ASM'
.section .rodata
.p2align 4
.global arena_model_begin
arena_model_begin:
.incbin "/work/build-arena/model.bin"
.global arena_model_end
arena_model_end:
.section .note.GNU-stack,"",@progbits
ASM
docker run --rm --platform linux/amd64 \
  -v "$repo_dir:/work" -w /work alpine:latest sh -eu -c '
    apk add --no-cache g++ binutils >/dev/null
    g++ -std=c++17 -O2 -static -no-pie -DARENA_EMBED_MODEL \
      -I. -Iopen_spiel/json/include \
      open_spiel/examples/arena_hulhe_bot.cc build-arena/arena_model.S \
      -o build-arena/dist/arena_hulhe_bot
    strip build-arena/dist/arena_hulhe_bot
  '
