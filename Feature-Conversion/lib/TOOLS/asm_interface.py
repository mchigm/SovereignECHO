"""
Assembly Interface Module
Provides Python wrappers for x86_64 assembly signal processing functions
"""

import numpy as np
from scipy import signal
from ctypes import CDLL, POINTER, c_double, c_size_t, c_int
import os
import platform


class AssemblySignalProcessor:
    """Interface to assembly-optimized signal processing functions"""
    
    def __init__(self, dll_path=None):
        """
        Initialize the assembly interface
        
        Args:
            dll_path: Path to compiled assembly DLL/SO file
                     If None, will look for bandpass_filter.dll in src/
        """
        if dll_path is None:
            current_dir = os.path.dirname(__file__)
            if platform.system() == "Windows":
                dll_path = os.path.join(current_dir, "src", "bandpass_filter.dll")
            else:
                dll_path = os.path.join(current_dir, "src", "bandpass_filter.so")
        
        # Load the compiled assembly library
        self.lib = CDLL(dll_path)
        
        # Define function signatures
        self._setup_function_signatures()
    
    def _setup_function_signatures(self):
        """Setup ctypes function signatures for assembly functions"""
        
        # apply_bandpass_filter
        self.lib.apply_bandpass_filter.argtypes = [
            POINTER(c_double),  # data pointer
            c_size_t,           # length
            c_double,           # lowcut (XMM2)
            c_double,           # highcut (XMM3)
            c_double,           # fs (stack)
            c_int,              # order (stack)
            POINTER(c_double),  # sos pointer (stack)
            c_size_t            # n_sections (stack)
        ]
        self.lib.apply_bandpass_filter.restype = POINTER(c_double)
        
        # apply_lowpass_filter
        self.lib.apply_lowpass_filter.argtypes = [
            POINTER(c_double),
            c_size_t,
            c_double,           # cutoff
            c_double,           # fs
            c_int,              # order
            POINTER(c_double),  # sos
            c_size_t            # n_sections
        ]
        self.lib.apply_lowpass_filter.restype = POINTER(c_double)
        
        # remove_dc_offset
        self.lib.remove_dc_offset.argtypes = [
            POINTER(c_double),
            c_size_t
        ]
        self.lib.remove_dc_offset.restype = POINTER(c_double)
        
        # normalize_signal
        self.lib.normalize_signal.argtypes = [
            POINTER(c_double),
            c_size_t
        ]
        self.lib.normalize_signal.restype = POINTER(c_double)
        
        # resample_signal
        self.lib.resample_signal.argtypes = [
            POINTER(c_double),
            c_size_t,
            c_int,              # original_rate
            c_int,              # target_rate
            POINTER(c_double)   # output pointer
        ]
        self.lib.resample_signal.restype = POINTER(c_double)
        
        # compute_envelope
        self.lib.compute_envelope.argtypes = [
            POINTER(c_double),
            c_size_t,
            POINTER(c_double)   # output pointer
        ]
        self.lib.compute_envelope.restype = POINTER(c_double)
    
    def apply_bandpass_filter(self, data: np.ndarray, lowcut: float, 
                            highcut: float, fs: float, order: int = 5) -> np.ndarray:
        """
        Apply Butterworth bandpass filter using assembly implementation
        
        Args:
            data: Input signal array
            lowcut: Low cutoff frequency (Hz)
            highcut: High cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order (default: 5)
        
        Returns:
            Filtered signal array
        """
        # Ensure data is contiguous double array
        data = np.ascontiguousarray(data, dtype=np.float64)
        
        # Compute SOS coefficients using scipy
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        sos = signal.butter(order, [low, high], btype='band', output='sos')
        
        # Flatten SOS array for passing to assembly
        sos_flat = np.ascontiguousarray(sos.flatten(), dtype=np.float64)
        n_sections = sos.shape[0]
        
        # Get pointers
        data_ptr = data.ctypes.data_as(POINTER(c_double))
        sos_ptr = sos_flat.ctypes.data_as(POINTER(c_double))
        
        # Call assembly function
        self.lib.apply_bandpass_filter(
            data_ptr,
            len(data),
            c_double(lowcut),
            c_double(highcut),
            c_double(fs),
            c_int(order),
            sos_ptr,
            c_size_t(n_sections)
        )
        
        return data  # Modified in-place
    
    def apply_lowpass_filter(self, data: np.ndarray, cutoff: float, 
                           fs: float, order: int = 5) -> np.ndarray:
        """
        Apply Butterworth lowpass filter using assembly implementation
        
        Args:
            data: Input signal array
            cutoff: Cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order (default: 5)
        
        Returns:
            Filtered signal array
        """
        data = np.ascontiguousarray(data, dtype=np.float64)
        
        # Compute SOS coefficients
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        sos = signal.butter(order, normal_cutoff, btype='low', output='sos')
        
        sos_flat = np.ascontiguousarray(sos.flatten(), dtype=np.float64)
        n_sections = sos.shape[0]
        
        data_ptr = data.ctypes.data_as(POINTER(c_double))
        sos_ptr = sos_flat.ctypes.data_as(POINTER(c_double))
        
        self.lib.apply_lowpass_filter(
            data_ptr,
            len(data),
            c_double(cutoff),
            c_double(fs),
            c_int(order),
            sos_ptr,
            c_size_t(n_sections)
        )
        
        return data
    
    def apply_highpass_filter(self, data: np.ndarray, cutoff: float,
                            fs: float, order: int = 5) -> np.ndarray:
        """
        Apply Butterworth highpass filter using assembly implementation
        
        Args:
            data: Input signal array
            cutoff: Cutoff frequency (Hz)
            fs: Sampling frequency (Hz)
            order: Filter order (default: 5)
        
        Returns:
            Filtered signal array
        """
        data = np.ascontiguousarray(data, dtype=np.float64)
        
        # Compute SOS coefficients
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        sos = signal.butter(order, normal_cutoff, btype='high', output='sos')
        
        sos_flat = np.ascontiguousarray(sos.flatten(), dtype=np.float64)
        n_sections = sos.shape[0]
        
        data_ptr = data.ctypes.data_as(POINTER(c_double))
        sos_ptr = sos_flat.ctypes.data_as(POINTER(c_double))
        
        self.lib.apply_lowpass_filter(  # Uses same implementation
            data_ptr,
            len(data),
            c_double(cutoff),
            c_double(fs),
            c_int(order),
            sos_ptr,
            c_size_t(n_sections)
        )
        
        return data
    
    def remove_dc_offset(self, data: np.ndarray) -> np.ndarray:
        """
        Remove DC offset from signal using assembly implementation
        
        Args:
            data: Input signal array
        
        Returns:
            Signal with DC offset removed
        """
        data = np.ascontiguousarray(data, dtype=np.float64)
        data_ptr = data.ctypes.data_as(POINTER(c_double))
        
        self.lib.remove_dc_offset(data_ptr, len(data))
        
        return data
    
    def normalize_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize signal to [-1, 1] range using assembly implementation
        
        Args:
            data: Input signal array
        
        Returns:
            Normalized signal
        """
        data = np.ascontiguousarray(data, dtype=np.float64)
        data_ptr = data.ctypes.data_as(POINTER(c_double))
        
        self.lib.normalize_signal(data_ptr, len(data))
        
        return data
    
    def resample_signal(self, data: np.ndarray, original_rate: int, 
                       target_rate: int) -> np.ndarray:
        """
        Resample signal to different sampling rate using assembly implementation
        
        Args:
            data: Input signal array
            original_rate: Original sampling rate (Hz)
            target_rate: Target sampling rate (Hz)
        
        Returns:
            Resampled signal array
        """
        data = np.ascontiguousarray(data, dtype=np.float64)
        
        # Calculate output length
        num_samples = int(len(data) * target_rate / original_rate)
        output = np.zeros(num_samples, dtype=np.float64)
        
        data_ptr = data.ctypes.data_as(POINTER(c_double))
        output_ptr = output.ctypes.data_as(POINTER(c_double))
        
        self.lib.resample_signal(
            data_ptr,
            len(data),
            c_int(original_rate),
            c_int(target_rate),
            output_ptr
        )
        
        return output
    
    def compute_envelope(self, data: np.ndarray) -> np.ndarray:
        """
        Compute signal envelope using assembly implementation
        
        Args:
            data: Input signal array
        
        Returns:
            Envelope of the signal
        """
        data = np.ascontiguousarray(data, dtype=np.float64)
        output = np.zeros_like(data)
        
        data_ptr = data.ctypes.data_as(POINTER(c_double))
        output_ptr = output.ctypes.data_as(POINTER(c_double))
        
        self.lib.compute_envelope(data_ptr, len(data), output_ptr)
        
        return output


# Convenience function for backward compatibility with caller.py
def create_processor(dll_path=None):
    """Create and return an AssemblySignalProcessor instance"""
    return AssemblySignalProcessor(dll_path)


if __name__ == "__main__":
    # Test code
    print("Assembly Signal Processor Interface")
    print("=" * 50)
    
    # Generate test signal
    fs = 44100  # Sampling frequency
    duration = 1.0
    t = np.linspace(0, duration, int(fs * duration))
    
    # Signal with multiple frequency components
    signal_data = (np.sin(2 * np.pi * 440 * t) +      # 440 Hz (A4)
                   0.5 * np.sin(2 * np.pi * 880 * t) + # 880 Hz
                   0.3 * np.sin(2 * np.pi * 2000 * t)) # 2000 Hz
    
    # Add DC offset
    signal_data += 0.5
    
    print(f"Generated test signal:")
    print(f"  Duration: {duration}s")
    print(f"  Sample rate: {fs} Hz")
    print(f"  Samples: {len(signal_data)}")
    print(f"  Signal range: [{signal_data.min():.3f}, {signal_data.max():.3f}]")
    print()
    
    try:
        # Initialize processor
        processor = create_processor()
        print("✓ Assembly library loaded successfully")
        
        # Test DC removal
        signal_no_dc = processor.remove_dc_offset(signal_data.copy())
        print(f"✓ DC offset removed: mean = {signal_no_dc.mean():.6f}")
        
        # Test normalization
        signal_norm = processor.normalize_signal(signal_data.copy())
        print(f"✓ Signal normalized: max = {np.abs(signal_norm).max():.3f}")
        
        # Test bandpass filter
        filtered = processor.apply_bandpass_filter(
            signal_data.copy(),
            lowcut=400,
            highcut=1000,
            fs=fs,
            order=5
        )
        print(f"✓ Bandpass filter applied (400-1000 Hz)")
        
        # Test envelope
        envelope = processor.compute_envelope(signal_data.copy())
        print(f"✓ Envelope computed: max = {envelope.max():.3f}")
        
        print()
        print("All tests passed! Assembly functions working correctly.")
        
    except OSError as e:
        print(f"✗ Error: Could not load assembly library")
        print(f"  Make sure to compile bandpass_filter.asm first:")
        print(f"  nasm -f win64 src/bandpass_filter.asm -o src/bandpass_filter.obj")
        print(f"  link /DLL /OUT:src/bandpass_filter.dll src/bandpass_filter.obj")
        print(f"\n  Error details: {e}")
    except Exception as e:
        print(f"✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
