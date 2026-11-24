#!python3
# Runs on any Python 3.9+. Dependencies: see requirements.txt (installed into .venv next to this file).
# The app itself lives in the quickimage package; this launcher only picks the interpreter and
# checks that required packages are installed before importing it.
from quickimage import bootstrap

bootstrap.ensure_venv(relaunch=__name__ == "__main__")

try:
    from quickimage.ui.app import main
except ImportError as _exc:
    _required = {d[0] for d in bootstrap.DEPENDENCIES if d[3]}
    if __name__ != "__main__" or (_exc.name or "").split(".")[0] not in _required:
        raise
    # pythonw has no console, so a missing package would otherwise fail silently
    bootstrap.startup_missing_packages(_exc)
    raise SystemExit(1)

if __name__ == "__main__":
    main()
