# Environment setup — what actually worked

Platform: Thunder Compute, L40 48GB (A6000 unavailable), 100GB disk

1. Python 3.10 via conda. The repo's requirements.txt pins 2022-era packages;
   `pathtools` cannot build on Python 3.12 because `imp` was removed.
   conda needed `--override-channels -c conda-forge` to bypass the
   Anaconda terms-of-service gate.
2. numpy pinned below 2.0 — the pinned packages predate numpy 2.
3. `pip install -e .` did not register. Used PYTHONPATH instead:
   export PYTHONPATH="$HOME/project/LIBERO-Safety:$PYTHONPATH"
4. System libraries needed, not listed anywhere in the repo:
   libgl1, libglib2.0-0, libegl1, libgles2, libglvnd0, libglew-dev
5. requirements.txt never completed in one pass; remaining packages
   installed one at a time until `from libero.libero import benchmark` worked.
6. MUJOCO_GL=egl for headless rendering. EGL throws harmless
   "Exception ignored in __del__" errors at interpreter shutdown.

Total: roughly two days.
