"""
Python wrapper for lutmm.c, the lookup-table matrix multiply. It compiles the C file on first
use (needs gcc or clang) and falls back to a slower NumPy version if that fails.
"""
import ctypes, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_SO = os.path.join(HERE, "_lutmm.so")
_lib = None


def _load():
    global _lib
    if _lib is not None:
        return _lib
    src = os.path.join(HERE, "lutmm.c")
    if not os.path.exists(_SO) or os.path.getmtime(_SO) < os.path.getmtime(src):
        for cc in ("gcc", "clang"):
            for flags in (["-fopenmp"], []):      # macOS clang has no OpenMP by default
                try:
                    subprocess.run([cc, "-O3", "-march=native", *flags, "-shared", "-fPIC", src, "-o", _SO],
                                   check=True, capture_output=True)
                    break
                except (OSError, subprocess.CalledProcessError):
                    continue
            if os.path.exists(_SO):
                break
    try:
        lib = ctypes.CDLL(_SO)
    except OSError:
        _lib = False
        return _lib
    P = np.ctypeslib.ndpointer
    lib.lut_matmul.argtypes = [P(np.uint8, flags="C"), P(np.uint8, flags="C"), P(np.int8, flags="C"),
                               P(np.int32, flags="C"), P(np.int32, flags="C"),
                               ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    _lib = lib
    return _lib


def lut_matmul(xq, mag, sgn, lut):
    """xq (N, K) uint8, mag (K, J) uint8, sgn (K, J) int8, lut (256, 256) int32 -> (N, J) int64."""
    xq = np.ascontiguousarray(xq, np.uint8)
    mag = np.ascontiguousarray(mag, np.uint8)
    sgn = np.ascontiguousarray(sgn, np.int8)
    lut = np.ascontiguousarray(lut, np.int32)
    N, K = xq.shape
    J = mag.shape[1]
    lib = _load()
    if lib:
        out = np.empty((N, J), np.int32)
        lib.lut_matmul(xq, mag, sgn, lut, out, N, K, J, int(not lut[0].any()))
        return out.astype(np.int64)
    out = np.empty((N, J), np.int64)                       # NumPy fallback
    s = sgn.astype(np.int64)
    for i in range(0, N, 64):
        out[i:i + 64] = (lut[xq[i:i + 64, :, None], mag[None]] * s[None]).sum(1)
    return out
