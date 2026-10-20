#!/bin/bash
# Run from the repo root: bash ocean/drone_fleet/tests/run_all.sh
set -e
if [ "$(uname -s)" = "Darwin" ]; then
    RAYLIB=raylib-5.5_macos
    LIBS=(-framework Cocoa -framework IOKit -framework CoreVideo -framework OpenGL)
else
    RAYLIB=raylib-5.5_linux_amd64
    LIBS=(-lGL)
fi
mkdir -p build/drone_fleet-tests
cc -std=gnu11 -O2 -Wall -Wextra -Wno-unused-parameter -Wno-unused-function -Wno-missing-braces \
    -I$RAYLIB/include -Isrc -Ivendor -Iocean/drone_fleet -DPLATFORM_DESKTOP \
    ocean/drone_fleet/tests/test_drone_fleet.c $RAYLIB/lib/libraylib.a \
    "${LIBS[@]}" -lm -lpthread -o build/drone_fleet-tests/test_drone_fleet
./build/drone_fleet-tests/test_drone_fleet
