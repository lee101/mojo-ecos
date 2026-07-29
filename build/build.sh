#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$root_dir/dist"
mojo build --emit shared-lib "$root_dir/src/ecos.mojo" -o "$root_dir/dist/libmojo-ecos.so"
