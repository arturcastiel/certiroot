"""Builds csrc/mas.cpp into certiroot/_libmas (a plain shared library loaded with ctypes).

Environment variables:
  GMP_DIR            prefix containing include/ and lib/ (default: system paths, Homebrew on macOS)
  GMP_INCLUDE_DIRS   extra include directories, os.pathsep-separated
  GMP_LIBRARY        full path of the GMP library to link instead of -lgmp
  CERTIROOT_ARCH     "native" (default: tuned for the build machine), "none" (compiler default,
                     portable), or any -march value such as "x86-64-v3"
"""
import os
import platform
import shutil
import subprocess
import sys
import tempfile

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.errors import CompileError


def _gmp_paths():
    inc, lib, objs = [], [], []
    prefixes = [os.environ["GMP_DIR"]] if os.environ.get("GMP_DIR") else []
    if not prefixes and sys.platform == "darwin" and shutil.which("brew"):
        try:
            prefixes = [subprocess.check_output(["brew", "--prefix", "gmp"], text=True).strip()]
        except (OSError, subprocess.CalledProcessError):
            pass
    for p in prefixes:
        inc.append(os.path.join(p, "include"))
        lib.append(os.path.join(p, "lib"))
    inc += [d for d in os.environ.get("GMP_INCLUDE_DIRS", "").split(os.pathsep) if d]
    if os.environ.get("GMP_LIBRARY"):
        objs.append(os.environ["GMP_LIBRARY"])
    return inc, lib, objs


class BuildExt(build_ext):
    def _accepts(self, flag):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "t.cpp")
            with open(src, "w") as f:
                f.write("int main() { return 0; }\n")
            try:
                self.compiler.compile([src], output_dir=d, extra_postargs=[flag])
                return True
            except CompileError:
                return False

    def build_extensions(self):
        if self.compiler.compiler_type == "msvc":
            raise RuntimeError("certiroot does not support MSVC yet (it needs __int128); use clang or gcc")
        arch = os.environ.get("CERTIROOT_ARCH", "native")
        flags = ["-O3", "-g0", "-std=c++17", "-ffp-contract=off"]
        if self._accepts("-fopenmp-simd"):
            flags.append("-fopenmp-simd")
        if arch == "native":
            flags += next(([f] for f in ("-march=native", "-mcpu=native") if self._accepts(f)), [])
        elif arch not in ("", "none"):
            flags.append("-march=" + arch)
        for ext in self.extensions:
            ext.extra_compile_args = flags + ext.extra_compile_args
        super().build_extensions()

    def get_ext_filename(self, ext_name):  # plain shared library: no Python ABI tag needed
        head, tail = os.path.split(super().get_ext_filename(ext_name))
        return os.path.join(head, tail.split(".")[0] + os.path.splitext(tail)[1])


if sys.platform == "darwin":  # universal2 Pythons would also build the x86_64 slice against arm64 GMP
    os.environ.setdefault("ARCHFLAGS", "-arch " + platform.machine())
inc, lib, objs = _gmp_paths()
setup(
    ext_modules=[Extension(
        "certiroot._libmas",
        sources=["csrc/mas.cpp"],
        include_dirs=inc,
        library_dirs=lib,
        libraries=[] if objs else ["gmp"],
        extra_objects=objs,
        language="c++",
    )],
    cmdclass={"build_ext": BuildExt},
)
