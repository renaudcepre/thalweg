#!/usr/bin/env bash
# Installs the WASM toolchain, one-shot after a fresh clone (#138): the
# wasm32 target, wasm-pack, and wasm-opt (binaryen).
#
# Used to be two `brew install` lines, so it stopped dead on any Linux
# without Homebrew ("brew: command not found", 2026-09-03). brew stays the
# first choice where it exists (prebuilt, fast, upgradable); without it:
#   - wasm-pack comes from `cargo install`: crates.io only, which any
#     machine that builds the project can already reach. A few minutes,
#     one-shot.
#   - wasm-opt comes from the binaryen release pinned by the caller, the
#     same one CI pulls (.github/workflows/dist.yml). NOT the distro
#     package: apt ships binaryen 108 (2022), which strips the target
#     features rustc emits and produces a module that computes nothing
#     (release v0.8.0, #141). Anything >= 121 understands them
#     (`call-indirect-overlong`, binaryen CHANGELOG v121).
# The tarball is a glibc binary: it won't run on NixOS, and `nix shell`
# is the right tool there anyway, hence the early exit with the command.
#
# Usage: wasm-setup.sh <binaryen version>
set -euo pipefail

BINARYEN_VERSION="${1:?expected a binaryen version, e.g. 132}"
BIN_DIR="$HOME/.local/bin"
SHARE_DIR="${XDG_DATA_HOME:-$HOME/.local/share}"

have() { command -v "$1" >/dev/null 2>&1; }

need_wasm_pack=0; have wasm-pack || need_wasm_pack=1
need_wasm_opt=0;  have wasm-opt  || need_wasm_opt=1

if [ -e /etc/NIXOS ] && ! have brew && [ $((need_wasm_pack + need_wasm_opt)) -gt 0 ]; then
    pkgs=""
    [ $need_wasm_pack = 1 ] && pkgs="$pkgs nixpkgs#wasm-pack"
    [ $need_wasm_opt = 1 ]  && pkgs="$pkgs nixpkgs#binaryen"
    echo "✗ NixOS: prebuilt binaries don't run here, get the tools from nixpkgs:" >&2
    echo "    nix shell$pkgs" >&2
    echo "  then rerun: just wasm-setup" >&2
    exit 1
fi

rustup target add wasm32-unknown-unknown

if [ $need_wasm_pack = 1 ]; then
    if have brew; then
        brew install wasm-pack
    else
        echo "▸ wasm-pack: cargo install (a few minutes, one-shot)"
        cargo install wasm-pack
    fi
fi

if [ $need_wasm_opt = 1 ]; then
    if have brew; then
        brew install binaryen
    else
        case "$(uname -s)-$(uname -m)" in
            Linux-x86_64)  platform="x86_64-linux" ;;
            Linux-aarch64) platform="aarch64-linux" ;;
            Darwin-arm64)  platform="arm64-macos" ;;
            Darwin-x86_64) platform="x86_64-macos" ;;
            *)
                echo "✗ no prebuilt binaryen for $(uname -s)/$(uname -m); install wasm-opt >= 121 by hand:" >&2
                echo "    https://github.com/WebAssembly/binaryen/releases/tag/version_$BINARYEN_VERSION" >&2
                exit 1 ;;
        esac
        name="binaryen-version_$BINARYEN_VERSION"
        url="https://github.com/WebAssembly/binaryen/releases/download/version_$BINARYEN_VERSION/$name-$platform.tar.gz"
        echo "▸ wasm-opt: binaryen $BINARYEN_VERSION → $SHARE_DIR/$name"
        mkdir -p "$SHARE_DIR" "$BIN_DIR"
        curl -fsSL "$url" | tar xz -C "$SHARE_DIR"
        ln -sf "$SHARE_DIR/$name/bin/wasm-opt" "$BIN_DIR/wasm-opt"
        if ! "$BIN_DIR/wasm-opt" --version >/dev/null 2>&1; then
            rm -f "$BIN_DIR/wasm-opt"
            echo "✗ the prebuilt wasm-opt doesn't run on this system; install >= 121 by hand:" >&2
            echo "    https://github.com/WebAssembly/binaryen/releases/tag/version_$BINARYEN_VERSION" >&2
            exit 1
        fi
        case ":$PATH:" in
            *":$BIN_DIR:"*) ;;
            *)
                export PATH="$BIN_DIR:$PATH"
                echo "⚠ $BIN_DIR is not on your PATH, \`just wasm\` needs it:"
                echo "    export PATH=\"$BIN_DIR:\$PATH\"" ;;
        esac
    fi
fi

echo "✓ $(wasm-pack --version), $(wasm-opt --version)"
