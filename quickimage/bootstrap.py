"""Startup support that needs only the standard library: venv handling, the dependency list,
installing packages, and the Dependencies dialog (it must work when packages are missing)."""

import os
import queue
import site
import subprocess
import sys
import threading
import tkinter
import tkinter.font
from pathlib import Path
from tkinter import ttk

_here = Path(__file__).resolve().parent.parent  # project root (assets/, .venv/, quickimage.pyw)
_venv = _here / ".venv"
_venv_sp = _venv / "Lib" / "site-packages"
_RELAUNCH_ENV = "QUICKIMAGE_RELAUNCHED"
LAUNCHER = _here / "quickimage.pyw"

APP_NAME = "Quick Image"
APP_ICON_ICO = _here / "assets" / "quickimage.ico"
APP_ICON_PNG = _here / "assets" / "quickimage.png"
APP_USER_MODEL_ID = "QuickImage.App"  # Windows taskbar identity, so it shows our icon, not Python's

def _venv_python_version(venv):
    """Return (major, minor) from the venv's pyvenv.cfg, or None if unknown."""
    try:
        for line in (venv / "pyvenv.cfg").read_text(encoding="utf-8").splitlines():
            key, _, val = line.partition("=")
            if key.strip() in ("version", "version_info"):
                parts = val.strip().split(".")
                return int(parts[0]), int(parts[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def _running_in_venv(venv):
    try:
        return Path(sys.prefix).resolve() == venv.resolve()
    except OSError:
        return False

def set_app_user_model_id():
    """Give the process its own Windows taskbar identity and make it DPI-aware.

    Without the identity, Windows groups the window under pythonw.exe and shows Python's
    icon in the taskbar. Without DPI awareness, Windows bitmap-stretches the whole window
    on high-DPI screens, which makes all text blurry. Both must happen before Tk starts.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
    except Exception:
        pass
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # system DPI aware
    except Exception:
        pass


def set_window_icon(root):
    """Title bar / taskbar / Alt-Tab icon. Missing icon files are not fatal."""
    try:
        if sys.platform == "win32" and APP_ICON_ICO.is_file():
            root.iconbitmap(default=str(APP_ICON_ICO))  # .ico carries crisp small sizes
        elif APP_ICON_PNG.is_file():
            photo = tkinter.PhotoImage(file=str(APP_ICON_PNG))
            root.iconphoto(True, photo)
            root._app_icon = photo  # keep a reference so Tk does not drop the image
    except tkinter.TclError:
        pass


# --- dependencies (standard library only: this must work when packages are missing) -----------

DEPENDENCIES = (
    # (import name, pip package, what it is used for, required to start the app)
    ("numpy", "numpy", "Image math", True),
    ("PIL", "pillow", "Opening and saving images", True),
    ("scipy", "scipy", "Background removal and color cleanup", True),
    ("tkinterdnd2", "tkinterdnd2", "Drag and drop", True),
    ("cv2", "opencv-python-headless", "Object outline (cut out one object)", False),
    ("sv_ttk", "sv-ttk", "Modern Windows 11 look", False),
    ("darkdetect", "darkdetect", "Follow the Windows light / dark setting", False),
)
STATUS_OK = "#2E9E5B"
STATUS_MISSING = "#D64545"


def missing_dependencies(deps=None):
    """The DEPENDENCIES entries that the running interpreter cannot import."""
    import importlib
    import importlib.util

    importlib.invalidate_caches()  # see packages installed while the app is running
    missing = []
    for dep in DEPENDENCIES if deps is None else deps:
        try:
            found = importlib.util.find_spec(dep[0]) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            missing.append(dep)
    return missing


def venv_python(venv=_venv, windowed=False):
    if sys.platform == "win32":
        return venv / "Scripts" / ("pythonw.exe" if windowed else "python.exe")
    return venv / "bin" / "python"


def _console_python(exe):
    """python.exe next to pythonw.exe: pip and venv need a console interpreter."""
    p = Path(exe)
    if p.name.lower() == "pythonw.exe" and p.with_name("python.exe").is_file():
        return str(p.with_name("python.exe"))
    return str(p)


def install_commands(venv=_venv, deps=None):
    """Commands that make the project's .venv complete: create it if needed, then pip install.

    Everything goes into .venv (where the launcher runs the app from). All packages are
    listed; pip skips the ones already installed, and a brand-new venv gets everything.
    """
    deps = DEPENDENCIES if deps is None else deps
    cmds = []
    if not venv_python(venv).is_file():
        cmds.append([_console_python(sys.executable), "-m", "venv", str(venv)])
    cmds.append(
        [str(venv_python(venv)), "-m", "pip", "install", "--disable-pip-version-check", *[d[1] for d in deps]]
    )
    return cmds


def run_commands(cmds, on_line, on_done):
    """Run commands one after another (call from a worker thread), streaming their output."""
    flags = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW: no console flash
    try:
        for cmd in cmds:
            on_line("> " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=flags,
            )
            for line in proc.stdout:
                on_line(line.rstrip())
            code = proc.wait()
            if code != 0:
                on_done(False, f"{Path(cmd[0]).name} exited with code {code}")
                return
    except OSError as exc:
        on_done(False, str(exc))
        return
    on_done(True, None)


def restart_app():
    """Start a fresh copy of the app (under .venv when it exists); the caller then exits."""
    exe = venv_python(windowed=True)
    if not exe.is_file():
        exe = Path(sys.executable)
    env = {k: v for k, v in os.environ.items() if k != _RELAUNCH_ENV}
    subprocess.Popen([str(exe), str(LAUNCHER)], env=env)


class DependencyDialog:
    """Lists the packages Quick Image uses, shows which are missing, and installs them.

    Uses only tkinter/ttk, so it also works at startup when required packages are missing.
    """

    def __init__(self, parent, startup_error=None, on_restart=None):
        self.parent = parent
        self.on_restart = on_restart
        self.events = queue.Queue()
        self.top = top = tkinter.Toplevel(parent)
        top.title(f"{APP_NAME} — Dependencies")
        top.resizable(True, True)
        try:
            if parent.winfo_viewable():
                top.transient(parent)
        except tkinter.TclError:
            pass
        self.frame = frame = ttk.Frame(top, padding=18)
        frame.grid(sticky="nsew")
        top.columnconfigure(0, weight=1)
        top.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.missing = missing_dependencies()
        if startup_error:
            intro = (
                f"{APP_NAME} could not start because required packages are missing "
                f"({startup_error}). It can install them now into the .venv folder next to quickimage.pyw."
            )
        elif self.missing:
            intro = "Some packages are missing. Quick Image can install them into its .venv folder."
        else:
            intro = "Everything Quick Image uses is installed."
        ttk.Label(frame, text=intro, wraplength=520, justify="left").grid(row=0, column=0, sticky="w")

        # Fonts derived from the theme's label font, so they match it at any DPI
        body = tkinter.font.Font(top, font=ttk.Style(top).lookup("TLabel", "font") or "TkDefaultFont")
        self._head_font = body.copy()
        self._head_font.configure(weight="bold")
        self._log_font = tkinter.font.Font(top, family="Consolas" if sys.platform == "win32" else "TkFixedFont",
                                           size=body.actual("size"))

        table = ttk.Frame(frame)
        table.grid(row=1, column=0, sticky="ew", pady=(14, 10))
        table.columnconfigure(1, weight=1)
        for c, head in enumerate(("Package", "Used for", "Status")):
            ttk.Label(table, text=head, font=self._head_font).grid(
                row=0, column=c, sticky="w", padx=(0, 18), pady=(0, 4)
            )
        self.status_labels = {}
        missing_names = {d[0] for d in self.missing}
        for r, (mod, pip_name, purpose, required) in enumerate(DEPENDENCIES, start=1):
            ttk.Label(table, text=pip_name).grid(row=r, column=0, sticky="w", padx=(0, 18), pady=1)
            ttk.Label(table, text=purpose + ("" if required else " (optional)")).grid(
                row=r, column=1, sticky="w", padx=(0, 18), pady=1
            )
            is_missing = mod in missing_names
            lbl = ttk.Label(
                table,
                text=("Missing" if is_missing else "Installed"),
                foreground=STATUS_MISSING if is_missing else STATUS_OK,
            )
            lbl.grid(row=r, column=2, sticky="w", pady=1)
            self.status_labels[mod] = lbl

        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.log = tkinter.Text(frame, height=10, width=80, wrap="none", font=self._log_font, state="disabled")
        self.result = ttk.Label(frame, text="", wraplength=520, justify="left")

        buttons = ttk.Frame(frame)
        buttons.grid(row=5, column=0, sticky="e", pady=(12, 0))
        self.btn_install = ttk.Button(
            buttons,
            text=f"Install missing ({len(self.missing)})" if self.missing else "Reinstall all",
            command=self.install,
        )
        try:
            self.btn_install.configure(style="Accent.TButton")
        except tkinter.TclError:
            pass
        self.btn_restart = ttk.Button(buttons, text=f"Restart {APP_NAME}", command=self.restart)
        self.btn_close = ttk.Button(buttons, text="Close", command=self.close)
        self.btn_install.grid(row=0, column=0, padx=(0, 8))
        self.btn_close.grid(row=0, column=2)
        top.protocol("WM_DELETE_WINDOW", self.close)
        top.bind("<Escape>", lambda _e: self.close())
        self._installing = False

    def install(self):
        if self._installing:
            return
        self._installing = True
        self.btn_install.state(["disabled"])
        self.btn_close.state(["disabled"])
        self.progress.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.progress.start(12)
        self.log.grid(row=3, column=0, sticky="nsew")
        self.frame.rowconfigure(3, weight=1)
        self.result.configure(text="")
        cmds = install_commands()
        threading.Thread(
            target=run_commands,
            args=(cmds, lambda line: self.events.put(("line", line)), lambda ok, err: self.events.put(("done", (ok, err)))),
            name="quickimage-install",
            daemon=True,
        ).start()
        self.top.after(80, self._poll)

    def _poll(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "line":
                    self.log.configure(state="normal")
                    self.log.insert("end", value + "\n")
                    self.log.see("end")
                    self.log.configure(state="disabled")
                else:
                    self._finished(*value)
                    return
        except queue.Empty:
            pass
        self.top.after(80, self._poll)

    def _finished(self, ok, err):
        self._installing = False
        self.progress.stop()
        self.progress.grid_remove()
        self.btn_close.state(["!disabled"])
        still = missing_dependencies()
        missing_names = {d[0] for d in still}
        for mod, lbl in self.status_labels.items():
            gone = mod not in missing_names
            lbl.configure(
                text="Installed" if gone else "Missing", foreground=STATUS_OK if gone else STATUS_MISSING
            )
        self.result.grid(row=4, column=0, sticky="w", pady=(8, 0))
        if ok:
            self.result.configure(
                text=f"Done. Restart {APP_NAME} to use the new packages.", foreground=STATUS_OK
            )
            self.btn_restart.grid(row=0, column=1, padx=(0, 8))
            try:
                self.btn_restart.configure(style="Accent.TButton")
                self.btn_install.configure(style="TButton")
            except tkinter.TclError:
                pass
        else:
            self.result.configure(text=f"Installation failed: {err}. See the log above.", foreground=STATUS_MISSING)
            self.btn_install.state(["!disabled"])

    def restart(self):
        if self.on_restart is not None:
            self.on_restart()
        else:
            restart_app()
            self.close()

    def close(self):
        if self._installing:
            return
        self.top.destroy()



def ensure_venv(relaunch=False):
    """Use the project's .venv: re-launch under its interpreter (once, when `relaunch`), or else
    borrow its site-packages when the interpreter versions match."""
    if not _venv_sp.is_dir() or _running_in_venv(_venv):
        return
    if relaunch and not os.environ.get(_RELAUNCH_ENV):
        exe = "pythonw.exe" if sys.executable.lower().endswith("w.exe") else "python.exe"
        venv_py = _venv / "Scripts" / exe
        if venv_py.is_file():
            env = dict(os.environ, **{_RELAUNCH_ENV: "1"})
            raise SystemExit(subprocess.call([str(venv_py), str(LAUNCHER), *sys.argv[1:]], env=env))
    # Borrowing the venv's packages is only safe when the interpreter versions match.
    if _venv_python_version(_venv) == sys.version_info[:2]:
        site.addsitedir(str(_venv_sp))


def startup_missing_packages(exc):
    """Required packages are missing: offer to install them instead of failing silently."""
    try:
        set_app_user_model_id()  # DPI awareness + taskbar identity, before the window exists
        root = tkinter.Tk()
        root.withdraw()
        root.title(APP_NAME)
        set_window_icon(root)
        try:
            import sv_ttk

            sv_ttk.set_theme("light")
        except Exception:
            pass
        root.update()  # let the theme finish applying (fonts, colors) before building the dialog
        dlg = DependencyDialog(root, startup_error=exc)
        root.wait_window(dlg.top)
        root.destroy()
    except Exception:
        print(f"{APP_NAME} could not start: {exc}\nInstall the packages in requirements.txt.", file=sys.stderr)
