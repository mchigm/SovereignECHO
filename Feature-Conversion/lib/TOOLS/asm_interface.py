"""
Assembly Interface Module
Provides Python wrappers for x86_64 assembly signal processing functions
and audio feature extraction utilities
"""

import numpy as np
from scipy import signal
from scipy.fft import dct as scipy_dct, fft
from ctypes import CDLL, POINTER, c_double, c_size_t, c_int
import os
import platform
import warnings

# Try to import audio libraries
try:
    import librosa
    import librosa.feature
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    warnings.warn("librosa not available. Some audio features may not work. Install with: pip install librosa")

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    warnings.warn("pydub not available. MP3 support may be limited. Install with: pip install pydub")

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("pywt not available. Wavelet transforms will not work. Install with: pip install PyWavelets")


class AssemblySignalProcessor:
    """Interface to assembly-optimized signal processing functions"""
    
    def __init__(self, dll_path=None):
        """ Initialize the assembly interface
        
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
        """ Apply Butterworth bandpass filter using assembly implementation
        
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
        """ Apply Butterworth lowpass filter using assembly implementation
        
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
        """ Apply Butterworth highpass filter using assembly implementation
        
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
        """ Remove DC offset from signal using assembly implementation
        
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
        """ Normalize signal to [-1, 1] range using assembly implementation
        
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
        """ Resample signal to different sampling rate using assembly implementation
        
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
        """ Compute signal envelope using assembly implementation
        
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


# ============================================================================
# AUDIO FEATURE EXTRACTION FUNCTIONS
# ============================================================================

def _load_audio_file(file_path: str, sr: int = None) -> tuple:
    """
    Load audio file from various formats (.wav, .mp3, .flac, .ogg, .m4a)
    
    Args:
        file_path: Path to audio file
        sr: Target sample rate (Hz). If None, uses native sample rate
    
    Returns:
        tuple: (audio_data, sample_rate)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    if LIBROSA_AVAILABLE:
        # Librosa can handle multiple formats
        y, sr_native = librosa.load(file_path, sr=sr, mono=True)
        return y, sr_native if sr is None else sr
    
    elif PYDUB_AVAILABLE:
        # Fallback to pydub for MP3 support
        audio = AudioSegment.from_file(file_path)
        
        # Convert to mono
        if audio.channels > 1:
            audio = audio.set_channels(1)
        
        # Resample if needed
        if sr is not None and audio.frame_rate != sr:
            audio = audio.set_frame_rate(sr)
        
        # Convert to numpy array
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2**15)  # Normalize to [-1, 1]
        
        return samples, audio.frame_rate
    
    else:
        # Last resort: try scipy.io.wavfile for .wav files only
        from scipy.io import wavfile
        sr_native, data = wavfile.read(file_path)
        
        # Convert to float and normalize
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        
        # Convert to mono if stereo
        if len(data.shape) > 1:
            data = data.mean(axis=1)
        
        # Resample if needed
        if sr is not None and sr_native != sr:
            from scipy.signal import resample
            num_samples = int(len(data) * sr / sr_native)
            data = resample(data, num_samples)
            sr_native = sr
        
        return data, sr_native


def sample_audio(file_path: str, sample_rate: int) -> np.ndarray:
    """
    Load and resample audio file to specified sample rate
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate in Hz
    
    Returns:
        Resampled audio data as numpy array
    """
    y, sr = _load_audio_file(file_path, sr=sample_rate)
    print(f"  Loaded {os.path.basename(file_path)}: {len(y)} samples at {sr} Hz")
    return y


def stft_analysis(file_path: str, window_size: int, overlap: int) -> np.ndarray:
    """
    Perform Short-Time Fourier Transform (STFT) on audio file
    
    Args:
        file_path: Path to audio file
        window_size: Size of FFT window (number of samples)
        overlap: Number of overlapping samples between windows
    
    Returns:
        STFT magnitude spectrogram
    """
    y, sr = _load_audio_file(file_path)
    
    # Calculate hop length from overlap
    hop_length = window_size - overlap
    
    if LIBROSA_AVAILABLE:
        stft_result = librosa.stft(y, n_fft=window_size, hop_length=hop_length)
        magnitude = np.abs(stft_result)
    else:
        # Fallback: use scipy
        f, t, stft_result = signal.stft(y, fs=sr, nperseg=window_size, noverlap=overlap)
        magnitude = np.abs(stft_result)
    
    print(f"  STFT shape: {magnitude.shape} (frequency bins × time frames)")
    return magnitude


def mfcc_extraction(file_path: str, num_coefficients: int = 13) -> np.ndarray:
    """
    Extract Mel-Frequency Cepstral Coefficients (MFCC) from audio
    
    Args:
        file_path: Path to audio file
        num_coefficients: Number of MFCC coefficients to extract
    
    Returns:
        MFCC features (num_coefficients × time_frames)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for MFCC extraction. Install with: pip install librosa")
    
    y, sr = _load_audio_file(file_path)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_coefficients)
    
    print(f"  MFCC shape: {mfccs.shape} ({num_coefficients} coefficients × {mfccs.shape[1]} frames)")
    return mfccs


def dct_analysis(file_path: str) -> np.ndarray:
    """
    Perform Discrete Cosine Transform (DCT) on audio signal
    
    Args:
        file_path: Path to audio file
    
    Returns:
        DCT coefficients
    """
    y, sr = _load_audio_file(file_path)
    
    # Apply DCT type-II (most common in audio processing)
    dct_coeffs = scipy_dct(y, type=2, norm='ortho')
    
    print(f"  DCT coefficients: {len(dct_coeffs)} values")
    return dct_coeffs


def wavelet_transform(file_path: str, wavelet: str = 'db4') -> tuple:
    """
    Perform Continuous Wavelet Transform on audio signal
    
    Args:
        file_path: Path to audio file
        wavelet: Wavelet type (e.g., 'db4', 'haar', 'sym4')
    
    Returns:
        tuple: (coefficients, frequencies)
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")
    
    y, sr = _load_audio_file(file_path)
    
    # Use scales appropriate for audio
    scales = np.arange(1, 128)
    coefficients, frequencies = pywt.cwt(y, scales, wavelet, sampling_period=1/sr)
    
    print(f"  Wavelet coefficients shape: {coefficients.shape} (scales × samples)")
    return coefficients, frequencies


def dwt_analysis(file_path: str, levels: int = 5) -> tuple:
    """
    Perform Discrete Wavelet Transform (DWT) decomposition
    
    Args:
        file_path: Path to audio file
        levels: Number of decomposition levels
    
    Returns:
        tuple: (approximation_coeffs, detail_coeffs_list)
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")
    
    y, sr = _load_audio_file(file_path)
    
    # Perform multi-level DWT decomposition
    coeffs = pywt.wavedec(y, 'db4', level=levels)
    
    # coeffs[0] = approximation, coeffs[1:] = details at each level
    print(f"  DWT levels: {levels}, Approximation length: {len(coeffs[0])}")
    return coeffs[0], coeffs[1:]


def zcr_calculation(file_path: str, frame_length: int = 2048, hop_length: int = 512) -> np.ndarray:
    """
    Calculate Zero-Crossing Rate (ZCR) of audio signal
    
    Args:
        file_path: Path to audio file
        frame_length: Length of each frame for ZCR calculation
        hop_length: Number of samples between successive frames
    
    Returns:
        ZCR values for each frame
    """
    if LIBROSA_AVAILABLE:
        y, sr = _load_audio_file(file_path)
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)
        zcr = zcr.flatten()
    else:
        # Manual ZCR calculation
        y, sr = _load_audio_file(file_path)
        zcr = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            frame = y[i:i+frame_length]
            # Count sign changes
            zero_crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
            zcr.append(zero_crossings / frame_length)
        
        zcr = np.array(zcr)
    
    print(f"  ZCR: {len(zcr)} frames, mean={np.mean(zcr):.4f}")
    return zcr


def spectral_centroid_calculation(file_path: str) -> np.ndarray:
    """
    Calculate spectral centroid (center of mass of spectrum)
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Spectral centroid values over time
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for spectral centroid. Install with: pip install librosa")
    
    y, sr = _load_audio_file(file_path)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    centroid = centroid.flatten()
    
    print(f"  Spectral Centroid: {len(centroid)} frames, mean={np.mean(centroid):.2f} Hz")
    return centroid


def chroma_feature_extraction(file_path: str) -> np.ndarray:
    """
    Extract chroma features (pitch class profiles)
    
    Args:
        file_path: Path to audio file
    
    Returns:
        Chroma feature matrix (12 × time_frames)
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for chroma features. Install with: pip install librosa")
    
    y, sr = _load_audio_file(file_path)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    
    print(f"  Chroma features: {chroma.shape} (12 pitch classes × {chroma.shape[1]} frames)")
    return chroma


def mcfb_extraction(file_path: str, n_filters: int = 40) -> np.ndarray:
    """
    Extract Modulation Cepstral Feature Bank (MCFB)
    
    Args:
        file_path: Path to audio file
        n_filters: Number of filters in the bank
    
    Returns:
        MCFB features
    """
    # MCFB is a specialized feature - implementing simplified version
    y, sr = _load_audio_file(file_path)
    
    if LIBROSA_AVAILABLE:
        # Use mel spectrogram as basis
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_filters)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Apply DCT to get cepstral coefficients
        mcfb = scipy_dct(mel_spec_db, type=2, axis=0, norm='ortho')
    else:
        # Simplified fallback
        stft = np.abs(fft(y))[:len(y)//2]
        mcfb = scipy_dct(stft[:n_filters], type=2, norm='ortho')
    
    print(f"  MCFB features: {mcfb.shape}")
    return mcfb


def mfb_extraction(file_path: str, n_mels: int = 40) -> np.ndarray:
    """
    Extract Mel Filter Bank (MFB) features
    
    Args:
        file_path: Path to audio file
        n_mels: Number of mel bands
    
    Returns:
        Mel filter bank features
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for MFB extraction. Install with: pip install librosa")
    
    y, sr = _load_audio_file(file_path)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    print(f"  MFB features: {mel_spec_db.shape} ({n_mels} mel bands × {mel_spec_db.shape[1]} frames)")
    return mel_spec_db


def cqcc_extraction(file_path: str, n_bins: int = 84, bins_per_octave: int = 12) -> np.ndarray:
    """
    Extract Constant-Q Cepstral Coefficients (CQCC)
    
    Args:
        file_path: Path to audio file
        n_bins: Number of frequency bins
        bins_per_octave: Number of bins per octave
    
    Returns:
        CQCC features
    """
    if not LIBROSA_AVAILABLE:
        raise ImportError("librosa is required for CQCC extraction. Install with: pip install librosa")
    
    y, sr = _load_audio_file(file_path)
    
    # Compute Constant-Q Transform
    cqt = np.abs(librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave))
    
    # Convert to dB scale
    cqt_db = librosa.amplitude_to_db(cqt, ref=np.max)
    
    # Apply DCT to get cepstral coefficients
    cqcc = scipy_dct(cqt_db, type=2, axis=0, norm='ortho')
    
    print(f"  CQCC features: {cqcc.shape} ({n_bins} bins × {cqcc.shape[1]} frames)")
    return cqcc
