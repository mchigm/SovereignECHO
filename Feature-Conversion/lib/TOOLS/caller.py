import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
from typing import Tuple, Optional, Union
import os
import platform
import subprocess
from ctypes import CDLL, POINTER, c_double, c_size_t, c_int


# =============================================================================
# ASSEMBLY INTERFACE - Load compiled assembly functions if available
# =============================================================================

class AssemblyBackend:
    """Loads and interfaces with compiled x86_64 assembly functions"""
    
    def __init__(self):
        self.lib = None
        self.available = False
        self._try_load_assembly()
    
    def _try_load_assembly(self):
        """Attempt to load the compiled assembly library"""
        try:
            current_dir = os.path.dirname(__file__)
            if platform.system() == "Windows":
                lib_path = os.path.join(current_dir, "src", "bandpass_filter.dll")
            else:
                lib_path = os.path.join(current_dir, "src", "bandpass_filter.so")
            
            if os.path.exists(lib_path):
                self.lib = CDLL(lib_path)
                self._setup_signatures()
                self.available = True
            else:
                self.available = False
        except Exception:
            self.available = False
    
    def _setup_signatures(self):
        """Setup ctypes function signatures"""
        # apply_bandpass_filter
        self.lib.apply_bandpass_filter.argtypes = [
            POINTER(c_double), c_size_t, c_double, c_double,
            c_double, c_int, POINTER(c_double), c_size_t
        ]
        self.lib.apply_bandpass_filter.restype = POINTER(c_double)
        
        # apply_lowpass_filter
        self.lib.apply_lowpass_filter.argtypes = [
            POINTER(c_double), c_size_t, c_double, c_double,
            c_int, POINTER(c_double), c_size_t
        ]
        self.lib.apply_lowpass_filter.restype = POINTER(c_double)
        
        # remove_dc_offset
        self.lib.remove_dc_offset.argtypes = [POINTER(c_double), c_size_t]
        self.lib.remove_dc_offset.restype = POINTER(c_double)
        
        # normalize_signal
        self.lib.normalize_signal.argtypes = [POINTER(c_double), c_size_t]
        self.lib.normalize_signal.restype = POINTER(c_double)
        
        # resample_signal
        self.lib.resample_signal.argtypes = [
            POINTER(c_double), c_size_t, c_int, c_int, POINTER(c_double)
        ]
        self.lib.resample_signal.restype = POINTER(c_double)
        
        # compute_envelope
        self.lib.compute_envelope.argtypes = [
            POINTER(c_double), c_size_t, POINTER(c_double)
        ]
        self.lib.compute_envelope.restype = POINTER(c_double)


# Global assembly backend instance
_asm_backend = AssemblyBackend()


def is_assembly_available() -> bool:
    """Check if assembly backend is available"""
    return _asm_backend.available


def build_assembly() -> bool:
    """
    Build the assembly library from source
    Returns True if successful, False otherwise
    """
    print("Building assembly library...")
    
    current_dir = os.path.dirname(__file__)
    asm_file = os.path.join(current_dir, "src", "bandpass_filter.asm")
    
    if not os.path.exists(asm_file):
        print(f"Error: Assembly source not found: {asm_file}")
        return False
    
    if platform.system() == "Windows":
        return _build_windows(current_dir, asm_file)
    elif platform.system() == "Linux":
        return _build_linux(current_dir, asm_file)
    else:
        print(f"Unsupported platform: {platform.system()}")
        return False


def _build_windows(current_dir: str, asm_file: str) -> bool:
    """Build for Windows x64"""
    obj_file = os.path.join(current_dir, "src", "bandpass_filter.obj")
    dll_file = os.path.join(current_dir, "src", "bandpass_filter.dll")
    
    # Assemble with NASM
    try:
        result = subprocess.run(
            ['nasm', '-f', 'win64', asm_file, '-o', obj_file],
            capture_output=True, text=True, check=True
        )
        print(f"✓ Assembled: {obj_file}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ Assembly failed: {e}")
        print("Install NASM from https://www.nasm.us/")
        return False
    
    # Link with Microsoft linker
    try:
        result = subprocess.run(
            ['link', '/DLL', f'/OUT:{dll_file}', obj_file, 
             'msvcrt.lib', 'legacy_stdio_definitions.lib'],
            capture_output=True, text=True, check=True
        )
        print(f"✓ Linked: {dll_file}")
        
        # Reload the backend
        global _asm_backend
        _asm_backend = AssemblyBackend()
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ Linking failed: {e}")
        print("Run from Visual Studio Developer Command Prompt")
        return False


def _build_linux(current_dir: str, asm_file: str) -> bool:
    """Build for Linux x64"""
    obj_file = os.path.join(current_dir, "src", "bandpass_filter.o")
    so_file = os.path.join(current_dir, "src", "bandpass_filter.so")
    
    # Assemble
    try:
        subprocess.run(
            ['nasm', '-f', 'elf64', asm_file, '-o', obj_file],
            capture_output=True, text=True, check=True
        )
        print(f"✓ Assembled: {obj_file}")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ Assembly failed: {e}")
        return False
    
    # Link
    try:
        subprocess.run(
            ['gcc', '-shared', '-o', so_file, obj_file, '-lm'],
            capture_output=True, text=True, check=True
        )
        print(f"✓ Linked: {so_file}")
        
        # Reload the backend
        global _asm_backend
        _asm_backend = AssemblyBackend()
        
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"✗ Linking failed: {e}")
        return False


# =============================================================================
# PYTHON IMPLEMENTATIONS (Original functions - always available)
# =============================================================================

def apply_bandpass_filter(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 5,
    use_assembly: bool = True
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter to the signal.
    
    Args:
        data: Input signal array
        lowcut: Low cutoff frequency (Hz)
        highcut: High cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order (default: 5)
        use_assembly: Use assembly implementation if available (default: True)
    
    Returns:
        Filtered signal array
    """
    # Try assembly implementation first if requested
    if use_assembly and _asm_backend.available:
        try:
            return _apply_bandpass_filter_asm(data, lowcut, highcut, fs, order)
        except Exception:
            pass  # Fall back to Python implementation
    
    # Python implementation
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    return signal.sosfilt(sos, data)


def _apply_bandpass_filter_asm(
    data: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int
) -> np.ndarray:
    """Assembly implementation of bandpass filter"""
    data = np.ascontiguousarray(data, dtype=np.float64)
    
    # Compute SOS coefficients
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    sos_flat = np.ascontiguousarray(sos.flatten(), dtype=np.float64)
    
    # Call assembly
    data_ptr = data.ctypes.data_as(POINTER(c_double))
    sos_ptr = sos_flat.ctypes.data_as(POINTER(c_double))
    
    _asm_backend.lib.apply_bandpass_filter(
        data_ptr, len(data),
        c_double(lowcut), c_double(highcut),
        c_double(fs), c_int(order),
        sos_ptr, c_size_t(sos.shape[0])
    )
    
    return data


def apply_lowpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 5,
    use_assembly: bool = True
) -> np.ndarray:
    """
    Apply a Butterworth lowpass filter to the signal.
    
    Args:
        data: Input signal array
        cutoff: Cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order (default: 5)
        use_assembly: Use assembly implementation if available (default: True)
    
    Returns:
        Filtered signal array
    """
    # Try assembly implementation
    if use_assembly and _asm_backend.available:
        try:
            return _apply_lowpass_filter_asm(data, cutoff, fs, order)
        except Exception:
            pass
    
    # Python implementation
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = signal.butter(order, normal_cutoff, btype='low', output='sos')
    return signal.sosfilt(sos, data)


def _apply_lowpass_filter_asm(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int
) -> np.ndarray:
    """Assembly implementation of lowpass filter"""
    data = np.ascontiguousarray(data, dtype=np.float64)
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = signal.butter(order, normal_cutoff, btype='low', output='sos')
    sos_flat = np.ascontiguousarray(sos.flatten(), dtype=np.float64)
    
    data_ptr = data.ctypes.data_as(POINTER(c_double))
    sos_ptr = sos_flat.ctypes.data_as(POINTER(c_double))
    
    _asm_backend.lib.apply_lowpass_filter(
        data_ptr, len(data),
        c_double(cutoff), c_double(fs), c_int(order),
        sos_ptr, c_size_t(sos.shape[0])
    )
    
    return data


def apply_highpass_filter(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 5,
    use_assembly: bool = True
) -> np.ndarray:
    """
    Apply a Butterworth highpass filter to the signal.
    
    Args:
        data: Input signal array
        cutoff: Cutoff frequency (Hz)
        fs: Sampling frequency (Hz)
        order: Filter order (default: 5)
        use_assembly: Use assembly implementation if available (default: True)
    
    Returns:
        Filtered signal array
    """
    # Assembly highpass uses same implementation as lowpass (different SOS coefficients)
    if use_assembly and _asm_backend.available:
        try:
            return _apply_highpass_filter_asm(data, cutoff, fs, order)
        except Exception:
            pass
    
    # Python implementation
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = signal.butter(order, normal_cutoff, btype='high', output='sos')
    return signal.sosfilt(sos, data)


def _apply_highpass_filter_asm(
    data: np.ndarray,
    cutoff: float,
    fs: float,
    order: int
) -> np.ndarray:
    """Assembly implementation of highpass filter"""
    data = np.ascontiguousarray(data, dtype=np.float64)
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    sos = signal.butter(order, normal_cutoff, btype='high', output='sos')
    sos_flat = np.ascontiguousarray(sos.flatten(), dtype=np.float64)
    
    data_ptr = data.ctypes.data_as(POINTER(c_double))
    sos_ptr = sos_flat.ctypes.data_as(POINTER(c_double))
    
    _asm_backend.lib.apply_lowpass_filter(  # Same implementation
        data_ptr, len(data),
        c_double(cutoff), c_double(fs), c_int(order),
        sos_ptr, c_size_t(sos.shape[0])
    )
    
    return data


def compute_fft(data: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute FFT and return frequencies and magnitudes."""
    n = len(data)
    yf = fft(data)
    xf = fftfreq(n, 1 / fs)[:n // 2]
    magnitude = 2.0 / n * np.abs(yf[:n // 2])
    return xf, magnitude


def compute_spectrogram(
    data: np.ndarray,
    fs: float,
    window: str = 'hann',
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute spectrogram of the signal."""
    f, t, Sxx = signal.spectrogram(data, fs, window=window, nperseg=nperseg)
    return f, t, Sxx


def resample_signal(
    data: np.ndarray,
    original_rate: int,
    target_rate: int,
    use_assembly: bool = True
) -> np.ndarray:
    """
    Resample signal to a different sampling rate.
    
    Args:
        data: Input signal array
        original_rate: Original sampling rate (Hz)
        target_rate: Target sampling rate (Hz)
        use_assembly: Use assembly implementation if available (default: True)
    
    Returns:
        Resampled signal
    """
    if use_assembly and _asm_backend.available:
        try:
            data = np.ascontiguousarray(data, dtype=np.float64)
            num_samples = int(len(data) * target_rate / original_rate)
            output = np.zeros(num_samples, dtype=np.float64)
            
            data_ptr = data.ctypes.data_as(POINTER(c_double))
            output_ptr = output.ctypes.data_as(POINTER(c_double))
            
            _asm_backend.lib.resample_signal(
                data_ptr, len(data),
                c_int(original_rate), c_int(target_rate),
                output_ptr
            )
            return output
        except Exception:
            pass
    
    # Python implementation
    num_samples = int(len(data) * target_rate / original_rate)
    return signal.resample(data, num_samples)


def remove_dc_offset(data: np.ndarray, use_assembly: bool = True) -> np.ndarray:
    """
    Remove DC offset from signal.
    
    Args:
        data: Input signal array
        use_assembly: Use assembly implementation if available (default: True)
    
    Returns:
        Signal with DC offset removed
    """
    if use_assembly and _asm_backend.available:
        try:
            data = np.ascontiguousarray(data, dtype=np.float64)
            data_ptr = data.ctypes.data_as(POINTER(c_double))
            _asm_backend.lib.remove_dc_offset(data_ptr, len(data))
            return data
        except Exception:
            pass
    
    # Python implementation
    return data - np.mean(data)


def normalize_signal(data: np.ndarray, use_assembly: bool = True) -> np.ndarray:
    """
    Normalize signal to [-1, 1] range.
    
    Args:
        data: Input signal array
        use_assembly: Use assembly implementation if available (default: True)
    
    Returns:
        Normalized signal
    """
    if use_assembly and _asm_backend.available:
        try:
            data = np.ascontiguousarray(data, dtype=np.float64)
            data_ptr = data.ctypes.data_as(POINTER(c_double))
            _asm_backend.lib.normalize_signal(data_ptr, len(data))
            return data
        except Exception:
            pass
    
    # Python implementation
    max_val = np.max(np.abs(data))
    return data / max_val if max_val > 0 else data


def detect_peaks(
    data: np.ndarray,
    height: Optional[float] = None,
    distance: Optional[int] = None,
    prominence: Optional[float] = None
) -> Tuple[np.ndarray, dict]:
    """Detect peaks in the signal."""
    peaks, properties = signal.find_peaks(
        data,
        height=height,
        distance=distance,
        prominence=prominence
    )
    return peaks, properties


def smooth_signal(data: np.ndarray, window_len: int = 11, window: str = 'hanning') -> np.ndarray:
    """Smooth signal using a window function."""
    if window_len < 3:
        return data
    
    s = np.r_[data[window_len-1:0:-1], data, data[-2:-window_len-1:-1]]
    
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = getattr(np, window)(window_len)
    
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[(window_len//2):-(window_len//2)]


def compute_welch_psd(
    data: np.ndarray,
    fs: float,
    nperseg: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method."""
    f, psd = signal.welch(data, fs, nperseg=nperseg)
    return f, psd


def apply_notch_filter(
    data: np.ndarray,
    freq: float,
    fs: float,
    quality: float = 30.0
) -> np.ndarray:
    """Apply a notch filter to remove a specific frequency."""
    b, a = signal.iirnotch(freq, quality, fs)
    return signal.filtfilt(b, a, data)


def compute_envelope(data: np.ndarray, use_assembly: bool = False) -> np.ndarray:
    """
    Compute the envelope of a signal using Hilbert transform.
    
    Args:
        data: Input signal array
        use_assembly: Use assembly implementation if available (default: False)
                     Note: Assembly version uses simplified absolute value method
    
    Returns:
        Envelope of the signal
    """
    if use_assembly and _asm_backend.available:
        try:
            data = np.ascontiguousarray(data, dtype=np.float64)
            output = np.zeros_like(data)
            
            data_ptr = data.ctypes.data_as(POINTER(c_double))
            output_ptr = output.ctypes.data_as(POINTER(c_double))
            
            _asm_backend.lib.compute_envelope(data_ptr, len(data), output_ptr)
            return output
        except Exception:
            pass
    
    # Python implementation (uses proper Hilbert transform)
    analytic_signal = signal.hilbert(data)
    return np.abs(analytic_signal)


# =============================================================================
# TESTING AND BENCHMARKING
# =============================================================================

def test_assembly_functions():
    """Test assembly implementations against Python versions"""
    print("=" * 70)
    print("ASSEMBLY FUNCTION TESTS")
    print("=" * 70)
    print()
    
    if not is_assembly_available():
        print("⚠️  Assembly backend not available")
        print("Run build_assembly() to compile the assembly code")
        return False
    
    print("✓ Assembly backend loaded")
    print()
    
    # Generate test signal
    fs = 44100
    duration = 0.1
    t = np.linspace(0, duration, int(fs * duration))
    test_signal = (
        np.sin(2 * np.pi * 440 * t) +
        0.5 * np.sin(2 * np.pi * 880 * t) +
        0.3 * np.sin(2 * np.pi * 2000 * t)
    )
    test_signal += 0.5  # Add DC offset
    
    results = []
    
    # Test 1: DC Removal
    print("Test 1: DC Offset Removal")
    py_result = remove_dc_offset(test_signal.copy(), use_assembly=False)
    asm_result = remove_dc_offset(test_signal.copy(), use_assembly=True)
    diff = np.max(np.abs(py_result - asm_result))
    passed = diff < 1e-10
    results.append(("DC Removal", passed, diff))
    print(f"  Max difference: {diff:.2e} - {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    
    # Test 2: Normalization
    print("Test 2: Signal Normalization")
    py_result = normalize_signal(test_signal.copy(), use_assembly=False)
    asm_result = normalize_signal(test_signal.copy(), use_assembly=True)
    diff = np.max(np.abs(py_result - asm_result))
    passed = diff < 1e-10
    results.append(("Normalization", passed, diff))
    print(f"  Max difference: {diff:.2e} - {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    
    # Test 3: Bandpass Filter
    print("Test 3: Bandpass Filter")
    py_result = apply_bandpass_filter(test_signal.copy(), 300, 3000, fs, use_assembly=False)
    asm_result = apply_bandpass_filter(test_signal.copy(), 300, 3000, fs, use_assembly=True)
    diff = np.max(np.abs(py_result - asm_result))
    passed = diff < 1e-10
    results.append(("Bandpass Filter", passed, diff))
    print(f"  Max difference: {diff:.2e} - {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    
    # Test 4: Lowpass Filter
    print("Test 4: Lowpass Filter")
    py_result = apply_lowpass_filter(test_signal.copy(), 1000, fs, use_assembly=False)
    asm_result = apply_lowpass_filter(test_signal.copy(), 1000, fs, use_assembly=True)
    diff = np.max(np.abs(py_result - asm_result))
    passed = diff < 1e-10
    results.append(("Lowpass Filter", passed, diff))
    print(f"  Max difference: {diff:.2e} - {'✓ PASS' if passed else '✗ FAIL'}")
    print()
    
    # Summary
    print("=" * 70)
    passed_count = sum(1 for _, p, _ in results if p)
    print(f"SUMMARY: {passed_count}/{len(results)} tests passed")
    print("=" * 70)
    
    return all(p for _, p, _ in results)


def benchmark_assembly():
    """Benchmark assembly vs Python implementations"""
    import time
    
    print()
    print("=" * 70)
    print("PERFORMANCE BENCHMARK")
    print("=" * 70)
    print()
    
    if not is_assembly_available():
        print("Assembly backend not available")
        return
    
    durations = [0.1, 0.5, 1.0, 5.0]
    fs = 44100
    
    for duration in durations:
        t = np.linspace(0, duration, int(fs * duration))
        test_signal = np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 2000 * t)
        n_samples = len(test_signal)
        
        print(f"Signal: {duration}s ({n_samples:,} samples)")
        
        # Bandpass filter benchmark
        py_data = test_signal.copy()
        start = time.perf_counter()
        apply_bandpass_filter(py_data, 300, 3000, fs, use_assembly=False)
        py_time = time.perf_counter() - start
        
        asm_data = test_signal.copy()
        start = time.perf_counter()
        apply_bandpass_filter(asm_data, 300, 3000, fs, use_assembly=True)
        asm_time = time.perf_counter() - start
        
        speedup = py_time / asm_time if asm_time > 0 else 0
        print(f"  Bandpass: Python={py_time*1000:.2f}ms, Assembly={asm_time*1000:.2f}ms, Speedup={speedup:.2f}x")
        print()


# =============================================================================
# MAIN - Run tests when module is executed directly
# =============================================================================

if __name__ == "__main__":
    print("Signal Processing Module with Assembly Acceleration")
    print()
    
    # Check assembly availability
    if is_assembly_available():
        print("✓ Assembly backend available and loaded")
        print()
        
        # Run tests
        test_assembly_functions()
        
        # Run benchmarks
        benchmark_assembly()
    else:
        print("⚠️  Assembly backend not available")
        print()
        print("To enable assembly acceleration:")
        print("  1. Make sure bandpass_filter.asm exists in src/")
        print("  2. Run: build_assembly()")
        print("  3. Or manually compile:")
        print("     Windows: nasm -f win64 src/bandpass_filter.asm -o src/bandpass_filter.obj")
        print("              link /DLL /OUT:src/bandpass_filter.dll src/bandpass_filter.obj msvcrt.lib")
        print("     Linux:   nasm -f elf64 src/bandpass_filter.asm -o src/bandpass_filter.o")
        print("              gcc -shared -o src/bandpass_filter.so src/bandpass_filter.o -lm")
        print()
        print("Module will use Python implementations (slower but fully functional)")
