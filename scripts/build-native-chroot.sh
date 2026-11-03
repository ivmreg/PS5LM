#!/usr/bin/env bash
set -euo pipefail

rootfs=/var/tmp/ps5lm-native-rootfs-20261101
if ! swapon --show=NAME --noheadings | grep -Fxq /var/tmp/ps5lm-native-20261101.swap; then
  swapon /var/tmp/ps5lm-native-20261101.swap || true
fi

mkdir -p "$rootfs/work" "$rootfs/input" "$rootfs/output" "$rootfs/proc" "$rootfs/dev"
mountpoint -q "$rootfs/work" || mount --bind /var/lib/docker/volumes/ps5lm-native-20261101/_data "$rootfs/work"
mountpoint -q "$rootfs/input" || mount --bind /mnt/c/Users/Igor/PS5LM "$rootfs/input"
mountpoint -q "$rootfs/proc" || mount -t proc proc "$rootfs/proc"
mountpoint -q "$rootfs/dev" || mount --rbind /dev "$rootfs/dev"
cp -f /etc/resolv.conf "$rootfs/etc/resolv.conf" 2>/dev/null || true

# Sync probe.cpp and any modified source files from host workspace into docker work volume
cp -f /mnt/c/Users/Igor/PS5LM/ps5/app/probe.cpp /var/lib/docker/volumes/ps5lm-native-20261101/_data/PS5LM/ps5/app/probe.cpp
cp -f /mnt/c/Users/Igor/PS5LM/scripts/release-app.sh /var/lib/docker/volumes/ps5lm-native-20261101/_data/PS5LM/scripts/release-app.sh
cp -f /mnt/c/Users/Igor/PS5LM/build/package_deploy.py /var/lib/docker/volumes/ps5lm-native-20261101/_data/PS5LM/build/package_deploy.py

chroot "$rootfs" /bin/bash -lc '
  set -euo pipefail
  cd /work/PS5LM
  export PATH=/usr/lib/llvm-19/bin:$PATH
  export LLVM_CONFIG=/usr/bin/llvm-config-19
  export BUILD_JOBS=2
  export CMAKE_BUILD_PARALLEL_LEVEL=2
  export PS5_VULKAN_ROOT="$PWD/.deps/native-ci-src/PS5_Vulkan"
  export PS5_PAYLOAD_SDK="$PS5_VULKAN_ROOT/.deps/native/ps5-payload-sdk"
  export PS5_OPENGL_SDK="$PWD/third_party/ps5-homebrew-ui/.deps/ps5-opengl/ps5-opengl-sdk-1.0.0/sdk"
  export PS5LM_PREBUILT_UI=OFF

  echo "=== Running native build-app.sh ==="
  bash scripts/build-app.sh

  echo "=== Packaging Release and Deployment Archives ==="
  bash scripts/release-app.sh
  python3 /input/build/package_deploy.py

  echo "=== Copying artifacts to host /input ==="
  mkdir -p /input/build/app-ps5/title/PPSA99575 /input/build/release /input/build/deployment
  cp -rf build/app-ps5/title/PPSA99575/* /input/build/app-ps5/title/PPSA99575/
  cp -rf build/release/* /input/build/release/
  cp -rf build/deployment/* /input/build/deployment/
  echo "All native build and packaging completed successfully!"
'
