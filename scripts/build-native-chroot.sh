#!/usr/bin/env bash
# Local helper: build the PS5 title inside a prepared chroot.
#
# This is not used by CI and needs a rootfs and a Docker volume prepared on the
# host. Set the paths for your machine:
#
#   PS5LM_NATIVE_ROOTFS       chroot root (default: /var/tmp/ps5lm-native-rootfs)
#   PS5LM_NATIVE_WORK_VOLUME  host path of the Docker volume with the source
#   PS5LM_HOST_WORKSPACE      host checkout to copy into that volume
#                             (default: the repository root)
#   PS5LM_NATIVE_SWAP         optional swap file to enable
#   PS5LM_NATIVE_BUILD_JOBS   build parallelism (default: 2)
#
# Example:
#   PS5LM_NATIVE_WORK_VOLUME=/var/lib/docker/volumes/ps5lm/_data \
#       scripts/build-native-chroot.sh
set -euo pipefail

rootfs=${PS5LM_NATIVE_ROOTFS:-/var/tmp/ps5lm-native-rootfs}
work_volume=${PS5LM_NATIVE_WORK_VOLUME:?set PS5LM_NATIVE_WORK_VOLUME to the Docker volume path}
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
host_workspace=${PS5LM_HOST_WORKSPACE:-$(cd -- "$script_dir/.." && pwd)}
build_jobs=${PS5LM_NATIVE_BUILD_JOBS:-2}

if [[ -n "${PS5LM_NATIVE_SWAP:-}" && -f "$PS5LM_NATIVE_SWAP" ]]; then
    if ! swapon --show=NAME --noheadings | grep -Fxq "$PS5LM_NATIVE_SWAP"; then
        swapon "$PS5LM_NATIVE_SWAP"
    fi
fi

mkdir -p "$rootfs/work" "$rootfs/input" "$rootfs/output" "$rootfs/proc" "$rootfs/dev"
mountpoint -q "$rootfs/work" || mount --bind "$work_volume" "$rootfs/work"
mountpoint -q "$rootfs/input" || mount --bind "$host_workspace" "$rootfs/input"
mountpoint -q "$rootfs/proc" || mount -t proc proc "$rootfs/proc"
mountpoint -q "$rootfs/dev" || mount --rbind /dev "$rootfs/dev"
cp -f /etc/resolv.conf "$rootfs/etc/resolv.conf" 2>/dev/null || true

# Sync modified source files from the host workspace into the Docker volume.
cp -rf "$host_workspace/ps5/app/." "$work_volume/PS5LM/ps5/app/"
cp -f "$host_workspace/scripts/release-app.sh" "$work_volume/PS5LM/scripts/release-app.sh"
if [[ -f "$host_workspace/build/package_deploy.py" ]]; then
    cp -f "$host_workspace/build/package_deploy.py" "$work_volume/PS5LM/build/package_deploy.py"
fi

chroot "$rootfs" /bin/bash -lc "
  set -euo pipefail
  cd /work/PS5LM
  export PATH=/usr/lib/llvm-19/bin:\$PATH
  export LLVM_CONFIG=/usr/bin/llvm-config-19
  export BUILD_JOBS=$build_jobs
  export CMAKE_BUILD_PARALLEL_LEVEL=$build_jobs
  export PS5_VULKAN_ROOT=\"\$PWD/.deps/native-ci-src/PS5_Vulkan\"
  export PS5_PAYLOAD_SDK=\"\$PS5_VULKAN_ROOT/.deps/native/ps5-payload-sdk\"
  export PS5_OPENGL_SDK=\"\$PWD/third_party/ps5-homebrew-ui/.deps/ps5-opengl/ps5-opengl-sdk-1.0.0/sdk\"
  export PS5LM_PREBUILT_UI=OFF

  echo '=== Running native build-app.sh ==='
  bash scripts/build-app.sh

  echo '=== Packaging Release and Deployment Archives ==='
  bash scripts/release-app.sh
  python3 /input/build/package_deploy.py

  echo '=== Copying artifacts to host /input ==='
  mkdir -p /input/build/app-ps5/title/PPSA99575 /input/build/release /input/build/deployment
  cp -rf build/app-ps5/title/PPSA99575/* /input/build/app-ps5/title/PPSA99575/
  cp -rf build/release/* /input/build/release/
  cp -rf build/deployment/* /input/build/deployment/
  echo 'All native build and packaging completed successfully!'
"
