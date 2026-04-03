"""
Solver backend abstraction.

Two backends:
    - "juliac" (default): calls a precompiled shared library via ctypes.
      No Julia installation required.
    - "juliacall": calls MOI + GenOpt + HiGHS through juliacall.
      Requires Julia (installed lazily by juliacall on first use).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jumpy.model import Model


class Backend(ABC):
    """Abstract solver backend."""

    @abstractmethod
    def optimize(self, model: Model) -> list[float]:
        """Solve the model and return the solution vector."""
        ...


class JuliacBackend(Backend):
    """
    Default backend: calls a precompiled Julia shared library via ctypes.

    The library is built with juliac from:
        MOI + GenOpt + Bridges + HiGHS

    No Julia installation required.
    """

    def __init__(self):
        self._lib = None

    def _load_lib(self):
        if self._lib is not None:
            return
        import ctypes
        import importlib.resources
        # TODO: resolve platform-specific library path
        # For now, search standard locations
        import os
        lib_names = [
            "libjumpy_backend.so",
            "libjumpy_backend.dylib",
            "jumpy_backend.dll",
        ]
        for name in lib_names:
            for search_dir in [os.path.dirname(__file__), os.getcwd(), "/usr/local/lib"]:
                path = os.path.join(search_dir, name)
                if os.path.exists(path):
                    self._lib = ctypes.CDLL(path)
                    return
        raise FileNotFoundError(
            "Could not find the compiled JuMPy backend library.\n"
            "The juliac-compiled shared library (libjumpy_backend.so) is not installed.\n"
            "Either:\n"
            "  1. Install the pre-built wheel: pip install jumpy\n"
            "  2. Use the juliacall backend: jp.Model(backend='juliacall')\n"
        )

    def optimize(self, model: Model) -> list[float]:
        self._load_lib()
        data = model._serialize()
        # TODO: implement ctypes calls to the compiled library
        raise NotImplementedError(
            "juliac backend not yet compiled. "
            "Use jp.Model(backend='juliacall') for now."
        )


class JuliaCallBackend(Backend):
    """
    Optional backend: calls Julia directly through juliacall.

    Requires `pip install jumpy[juliacall]`. Julia is installed lazily
    by juliacall on first use if not already present.

    This backend has full flexibility — it can use any solver or MOI
    feature, not just what's compiled into the juliac library.
    """

    def __init__(self):
        self._jl = None

    def _init_julia(self):
        if self._jl is not None:
            return
        try:
            from juliacall import Main as jl
        except ImportError:
            raise ImportError(
                "juliacall is not installed.\n"
                "Install it with: pip install jumpy[juliacall]\n"
                "This will also install Julia automatically if needed."
            ) from None
        # Install and load Julia packages on first use
        jl.seval("using Pkg")
        for pkg in ["MathOptInterface", "HiGHS", "GenOpt"]:
            jl.seval(f"""
                if !haskey(Pkg.project().dependencies, "{pkg}")
                    Pkg.add("{pkg}")
                end
            """)
        jl.seval("import MathOptInterface as MOI")
        jl.seval("import GenOpt")
        jl.seval("import HiGHS")
        # TODO: load GenOpt once it's registered / available
        self._jl = jl

    def optimize(self, model: Model) -> list[float]:
        self._init_julia()
        jl = self._jl
        return self._build_and_solve(jl, model)

    def _build_and_solve(self, jl, model: Model) -> list[float]:
        from jumpy.bridge_juliacall import build_moi_model
        return build_moi_model(jl, model)


_BACKENDS = {
    "juliac": JuliacBackend,
    "juliacall": JuliaCallBackend,
}


def get_backend(name: str) -> Backend:
    cls = _BACKENDS.get(name)
    if cls is None:
        raise ValueError(
            f"Unknown backend '{name}'. Choose from: {list(_BACKENDS.keys())}"
        )
    return cls()
