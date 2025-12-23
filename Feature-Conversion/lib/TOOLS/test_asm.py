"""
Test script for assembly signal processing functions
Compares assembly output with Python/SciPy reference implementation
"""

import numpy as np
from scipy import signal
import time
import sys

# Import both Python and assembly implementations
from caller import (
    apply_bandpass_filter as py_bandpass,
    apply_lowpass_filter as py_lowpass,
    remove_dc_offset as py_remove_dc,
    normalize_signal as py_normalize
)

try:
    from asm_interface import create_processor
    ASM_AVAILABLE = True
except Exception as e:
    print(f"Warning: Assembly functions not available: {e}")
    ASM_AVAILABLE = False


def generate_test_signal(duration=1.0, fs=44100):
    """Generate a test signal with multiple frequency components"""
    t = np.linspace(0, duration, int(fs * duration))
    
    # Multi-frequency signal
    signal_data = (
        np.sin(2 * np.pi * 440 * t) +       # 440 Hz (A4)
        0.5 * np.sin(2 * np.pi * 880 * t) + # 880 Hz (A5)
        0.3 * np.sin(2 * np.pi * 2000 * t) + # 2 kHz
        0.2 * np.sin(2 * np.pi * 100 * t)   # 100 Hz
    )
    
    # Add some noise
    signal_data += 0.05 * np.random.randn(len(signal_data))
    
    return signal_data, t


def compare_outputs(py_result, asm_result, tolerance=1e-6):
    """Compare Python and assembly outputs"""
    if py_result.shape != asm_result.shape:
        return False, "Shape mismatch"
    
    max_diff = np.max(np.abs(py_result - asm_result))
    rms_diff = np.sqrt(np.mean((py_result - asm_result) ** 2))
    
    match = max_diff < tolerance
    info = f"Max diff: {max_diff:.2e}, RMS diff: {rms_diff:.2e}"
    
    return match, info


def test_dc_removal():
    """Test DC offset removal"""
    print("\n" + "=" * 60)
    print("TEST: DC Offset Removal")
    print("=" * 60)
    
    # Generate signal with DC offset
    signal_data, _ = generate_test_signal(duration=0.1)
    signal_data += 2.5  # Add DC offset
    
    print(f"Input signal mean: {signal_data.mean():.6f}")
    
    # Python version
    py_data = signal_data.copy()
    start = time.time()
    py_result = py_remove_dc(py_data)
    py_time = time.time() - start
    print(f"Python result mean: {py_result.mean():.6f} (Time: {py_time*1000:.3f}ms)")
    
    # Assembly version
    if ASM_AVAILABLE:
        processor = create_processor()
        asm_data = signal_data.copy()
        start = time.time()
        asm_result = processor.remove_dc_offset(asm_data)
        asm_time = time.time() - start
        print(f"Assembly result mean: {asm_result.mean():.6f} (Time: {asm_time*1000:.3f}ms)")
        
        match, info = compare_outputs(py_result, asm_result)
        print(f"Match: {match} ({info})")
        if py_time > 0:
            print(f"Speedup: {py_time/asm_time:.2f}x")
        return match
    else:
        print("Assembly version not available")
        return None


def test_normalization():
    """Test signal normalization"""
    print("\n" + "=" * 60)
    print("TEST: Signal Normalization")
    print("=" * 60)
    
    signal_data, _ = generate_test_signal(duration=0.1)
    signal_data *= 5.0  # Scale up
    
    print(f"Input range: [{signal_data.min():.3f}, {signal_data.max():.3f}]")
    
    # Python version
    py_data = signal_data.copy()
    start = time.time()
    py_result = py_normalize(py_data)
    py_time = time.time() - start
    print(f"Python range: [{py_result.min():.3f}, {py_result.max():.3f}] (Time: {py_time*1000:.3f}ms)")
    
    # Assembly version
    if ASM_AVAILABLE:
        processor = create_processor()
        asm_data = signal_data.copy()
        start = time.time()
        asm_result = processor.normalize_signal(asm_data)
        asm_time = time.time() - start
        print(f"Assembly range: [{asm_result.min():.3f}, {asm_result.max():.3f}] (Time: {asm_time*1000:.3f}ms)")
        
        match, info = compare_outputs(py_result, asm_result)
        print(f"Match: {match} ({info})")
        if py_time > 0:
            print(f"Speedup: {py_time/asm_time:.2f}x")
        return match
    else:
        print("Assembly version not available")
        return None


def test_bandpass_filter():
    """Test bandpass filter"""
    print("\n" + "=" * 60)
    print("TEST: Bandpass Filter (300-3000 Hz)")
    print("=" * 60)
    
    fs = 44100
    signal_data, _ = generate_test_signal(duration=0.1, fs=fs)
    
    lowcut, highcut = 300, 3000
    order = 5
    
    print(f"Filter: {lowcut}-{highcut} Hz, Order: {order}")
    print(f"Signal length: {len(signal_data)} samples")
    
    # Python version
    py_data = signal_data.copy()
    start = time.time()
    py_result = py_bandpass(py_data, lowcut, highcut, fs, order)
    py_time = time.time() - start
    print(f"Python time: {py_time*1000:.3f}ms")
    
    # Assembly version
    if ASM_AVAILABLE:
        processor = create_processor()
        asm_data = signal_data.copy()
        start = time.time()
        asm_result = processor.apply_bandpass_filter(asm_data, lowcut, highcut, fs, order)
        asm_time = time.time() - start
        print(f"Assembly time: {asm_time*1000:.3f}ms")
        
        match, info = compare_outputs(py_result, asm_result, tolerance=1e-10)
        print(f"Match: {match} ({info})")
        if py_time > 0:
            print(f"Speedup: {py_time/asm_time:.2f}x")
        return match
    else:
        print("Assembly version not available")
        return None


def test_lowpass_filter():
    """Test lowpass filter"""
    print("\n" + "=" * 60)
    print("TEST: Lowpass Filter (1000 Hz)")
    print("=" * 60)
    
    fs = 44100
    signal_data, _ = generate_test_signal(duration=0.1, fs=fs)
    
    cutoff = 1000
    order = 5
    
    print(f"Filter: {cutoff} Hz cutoff, Order: {order}")
    
    # Python version
    py_data = signal_data.copy()
    start = time.time()
    py_result = py_lowpass(py_data, cutoff, fs, order)
    py_time = time.time() - start
    print(f"Python time: {py_time*1000:.3f}ms")
    
    # Assembly version
    if ASM_AVAILABLE:
        processor = create_processor()
        asm_data = signal_data.copy()
        start = time.time()
        asm_result = processor.apply_lowpass_filter(asm_data, cutoff, fs, order)
        asm_time = time.time() - start
        print(f"Assembly time: {asm_time*1000:.3f}ms")
        
        match, info = compare_outputs(py_result, asm_result, tolerance=1e-10)
        print(f"Match: {match} ({info})")
        if py_time > 0:
            print(f"Speedup: {py_time/asm_time:.2f}x")
        return match
    else:
        print("Assembly version not available")
        return None


def performance_benchmark():
    """Run performance benchmarks"""
    if not ASM_AVAILABLE:
        print("\nSkipping benchmark - assembly not available")
        return
    
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    durations = [0.1, 0.5, 1.0, 5.0]
    fs = 44100
    
    processor = create_processor()
    
    for duration in durations:
        signal_data, _ = generate_test_signal(duration=duration, fs=fs)
        n_samples = len(signal_data)
        
        print(f"\nSignal: {duration}s ({n_samples:,} samples)")
        
        # Bandpass filter benchmark
        py_data = signal_data.copy()
        start = time.time()
        py_bandpass(py_data, 300, 3000, fs, 5)
        py_time = time.time() - start
        
        asm_data = signal_data.copy()
        start = time.time()
        processor.apply_bandpass_filter(asm_data, 300, 3000, fs, 5)
        asm_time = time.time() - start
        
        print(f"  Bandpass: Python={py_time*1000:.2f}ms, Assembly={asm_time*1000:.2f}ms, Speedup={py_time/asm_time:.2f}x")


def main():
    """Run all tests"""
    print("=" * 60)
    print("ASSEMBLY SIGNAL PROCESSING TEST SUITE")
    print("=" * 60)
    
    if not ASM_AVAILABLE:
        print("\n⚠️  Assembly library not found!")
        print("Run build_asm.py first to compile the assembly code.")
        print("Testing Python implementation only...\n")
    
    results = []
    
    # Run tests
    results.append(("DC Removal", test_dc_removal()))
    results.append(("Normalization", test_normalization()))
    results.append(("Bandpass Filter", test_bandpass_filter()))
    results.append(("Lowpass Filter", test_lowpass_filter()))
    
    # Run benchmark
    if ASM_AVAILABLE:
        performance_benchmark()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        if result is None:
            status = "SKIPPED"
        elif result:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{test_name}: {status}")
    
    if ASM_AVAILABLE:
        passed = sum(1 for _, r in results if r == True)
        total = len([r for _, r in results if r is not None])
        print(f"\nPassed: {passed}/{total}")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
