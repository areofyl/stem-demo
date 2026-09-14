#!/bin/sh
# build WASM stem separator
# requires: emscripten (emsdk)

# main module (used by workers)
emcc inference_wasm.c \
  -O3 \
  -msimd128 \
  -ffast-math \
  -s WASM=1 \
  -s EXPORTED_FUNCTIONS='["_load_model_from_buffer","_separate","_get_output_ptr","_malloc","_free"]' \
  -s EXPORTED_RUNTIME_METHODS='["ccall","cwrap","HEAPF32","HEAPU8"]' \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s MODULARIZE=1 \
  -s EXPORT_NAME=StemModule \
  -s INITIAL_MEMORY=67108864 \
  -o stem.js

echo "built stem.js + stem.wasm"
