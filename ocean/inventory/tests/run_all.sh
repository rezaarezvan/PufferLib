#!/bin/bash
# Run from the repo root: bash ocean/inventory/tests/run_all.sh
set -e
if [ "$(uname -s)" = "Darwin" ]; then
    RAYLIB=raylib-5.5_macos
    LIBS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
else
    RAYLIB=raylib-5.5_linux_amd64
    LIBS=(-lGL)
fi
mkdir -p build/inventory-tests
cc -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -I$RAYLIB/include -Isrc -Ivendor -DPLATFORM_DESKTOP \
    ocean/inventory/tests/test_inventory.c $RAYLIB/lib/libraylib.a \
    "${LIBS[@]}" -lm -lpthread -o build/inventory-tests/test_inventory
./build/inventory-tests/test_inventory
