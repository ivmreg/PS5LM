# Driver patches

Four patches to PS5_Mesa's RADV, from [PS5CEMU-HAR](https://github.com/premohq/PS5CEMU-HAR)
(`patches/mesa`, GPL-3.0), applied on top of the Mesa revision PS5_Vulkan pins (`1ba9874`):

- `0007`: a destroyed buffer's memory waits until the GPU is done with it, so freed memory
  still in use cannot fault the GPU and stop the console. Always on.
- `0008`: submissions handed to a thread of the driver's own (`RADV_PS5_SUBMIT_THREAD=1`).
- `0009`: how long a wait for the GPU spins and sleeps (`RADV_PS5_WAIT_SPIN_US`,
  `RADV_PS5_WAIT_SLEEP_US`).
- `0010`: detects PS5 Pro (`CFI-70xx` / Viola) via `sceKernelGetHwModelName` (or `RADV_PS5_PRO=1`), updating topology to 60 CUs, 2.35 GHz boost clock, 2.25 GHz GDDR6 clock (576 GB/s bandwidth), and device marketing name to "PlayStation 5 Pro GPU". Optionally reports hardware integer dot product acceleration via `RADV_PS5_INT_DOT`:
  - `auto`: enabled on PS5 Pro (where RDNA 3 hardware executes integer dot product instructions); disabled on Base/Slim (Oberon GC 10.1.3, where hardware omits `v_dot4_*`).
  - `1` or `true`: forcefully enabled (warns on stderr if forced on non-Pro GC 10.1.3).
  - `0`, `false`, or unset: disabled (default).
  When enabled, ggml-vulkan activates its integer dot product (MMQ) compute shaders. Compute shaders bypass NGG workgroup repack, avoiding the graphics culling failure on Oberon.

0008, 0009 and 0010 change nothing unless asked. On the console 0008 and 0009 did not make generation faster
(2026-10-16, Qwen3.5 0.8B and 9B), so the app does not ask. Put the switches in
`/data/PS5LM/env.txt` to try them (e.g. `RADV_PS5_INT_DOT=auto` or `RADV_PS5_PRO=1`). To build RADV with the patches:

```sh
cd .deps/src/PS5_Mesa && git am ../../../patches/mesa/*.patch
(cd ../PS5_Vulkan && PS5_MESA_REVISION=$(git -C ../PS5_Mesa rev-parse HEAD) tools/build-radv.sh release)
```
