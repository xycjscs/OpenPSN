#!/bin/sh
# Build libpsn_chol.so with NO system C compiler present.
#
# Link recipe (the whole trick): the rustup toolchain ships `rust-lld`, which
# accepts rustc's GNU-flavoured linker arguments, but it has no default
# library search path, so the two -L flags below are mandatory.  The seven
# libX.so -> libX.so.N symlinks in ./linklibs exist because glibc only ships
# the versioned SONAMEs (libgcc_s.so.1, librt.so.1, ...) while rustc asks for
# the linker names (-lgcc_s, -lrt, ...).  Nothing under /usr is touched.
#
# Usage:  sh rust/build.sh
#         PSN_CHOL_CPU=native sh .../build.sh      # adds -C target-cpu=native
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
export PATH=${CARGO_HOME:-$HOME/.cargo}/bin:$PATH

SYSROOT=$(rustc --print sysroot)
LLD=$SYSROOT/lib/rustlib/$(rustc -Vv | sed -n "s/^host: //p")/bin/rust-lld
[ -x "$LLD" ] || { echo "rust-lld missing at $LLD" >&2; exit 1; }

CPU=${PSN_CHOL_CPU:-}
CPUFLAG=""
[ -n "$CPU" ] && CPUFLAG="-C target-cpu=$CPU"

# linker-name shims (idempotent)
mkdir -p "$HERE/linklibs"
for pair in "libc.so /usr/lib/x86_64-linux-gnu/libc.so" \
            "libm.so /usr/lib/x86_64-linux-gnu/libm.so" \
            "libgcc_s.so /lib/x86_64-linux-gnu/libgcc_s.so.1" \
            "librt.so /lib/x86_64-linux-gnu/librt.so.1" \
            "libpthread.so /lib/x86_64-linux-gnu/libpthread.so.0" \
            "libutil.so /lib/x86_64-linux-gnu/libutil.so.1" \
            "libdl.so /lib/x86_64-linux-gnu/libdl.so.2"; do
    set -- $pair
    [ -e "$HERE/linklibs/$1" ] || ln -s "$2" "$HERE/linklibs/$1"
done

exec rustc --edition 2021 \
    -C opt-level=3 -C codegen-units=1 -C debug-assertions=off \
    $CPUFLAG \
    --crate-type cdylib \
    --crate-name psn_chol \
    -C linker="$LLD" -C linker-flavor=ld.lld \
    -C link-arg=-L"$HERE/linklibs" \
    -C link-arg=-L/usr/lib/x86_64-linux-gnu \
    -C link-arg=-soname=libpsn_chol.so \
    -o "$HERE/libpsn_chol.so" \
    "$HERE/src/lib.rs"
