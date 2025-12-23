# Feature-Conversion: Song–Speech Conversion

A small toolkit to convert between melodic vocal features and speaker voice features, with optional x86_64 assembly acceleration for DSP hot paths.

---

This subproject accompanies our paper “Feature Engineering for Speaker Characterization in Melodic Vocalization” (IEEE ACAI 2025, DOI TBA). It provides Python utilities (NumPy/SciPy) and an assembly implementation of second‑order‑section (SOS) filtering for performance.

## Table of Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup](#setup)
4. [Quick Start](#quick-start)
5. [Assembly Acceleration (Optional)](#assembly-acceleration-optional)
6. [API Reference](#api-reference)
7. [Testing & Benchmarking](#testing--benchmarking)
8. [Questions](#questions)
9. [Academic](#academic)
10. [Contributing](#contributing)

## Overview

Core capabilities provided here:
- Bandpass/lowpass/highpass Butterworth filtering via SciPy, with an optional x86_64 SOS filter path in assembly.
- FFT helpers (frequencies + magnitudes), Welch PSD, spectrograms.
- Utility preprocessing: DC removal, normalization, peak detection, smoothing, envelope.

The Python entry points live in [Feature-Conversion/lib/TOOLS/caller.py](Feature-Conversion/lib/TOOLS/caller.py). Assembly routines are in [Feature-Conversion/lib/TOOLS/src/bandpass_filter.asm](Feature-Conversion/lib/TOOLS/src/bandpass_filter.asm).

## Project Structure

- [Feature-Conversion](Feature-Conversion)
     - [README.md](Feature-Conversion/README.md)
     - [src](Feature-Conversion/src)
          - [main.py](Feature-Conversion/src/main.py)
     - [lib](Feature-Conversion/lib)
          - [mod.pyw](Feature-Conversion/lib/mod.pyw)
          - [LIB](Feature-Conversion/lib/LIB)
               - [__init__.py](Feature-Conversion/lib/LIB/__init__.py)
               - [tools.py](Feature-Conversion/lib/LIB/tools.py)
          - [TOOLS](Feature-Conversion/lib/TOOLS)
               - [caller.py](Feature-Conversion/lib/TOOLS/caller.py)
               - [asm_interface.py](Feature-Conversion/lib/TOOLS/asm_interface.py)
               - [build_asm.py](Feature-Conversion/lib/TOOLS/build_asm.py)
               - [test_asm.py](Feature-Conversion/lib/TOOLS/test_asm.py)
               - [src](Feature-Conversion/lib/TOOLS/src)
                    - [bandpass_filter.asm](Feature-Conversion/lib/TOOLS/src/bandpass_filter.asm)
     - [Data](Feature-Conversion/Data)
          - [files](Feature-Conversion/Data/files)
          - [Folders](Feature-Conversion/Data/Folders)
          - [RESULT](Feature-Conversion/Data/RESULT)

## Setup

Minimal environment (Python-only path):

```powershell
# Create and activate a virtual environment (Conda shown)
conda create -n sovereignecho python=3.12 -y
conda activate sovereignecho

# Install required packages
pip install numpy scipy
```

Optional: install NASM if you plan to use the assembly path on Windows: https://www.nasm.us/

## Quick Start

Python usage with automatic assembly fallback (no extra steps required):

```python
import numpy as np
from Feature-Conversion.lib.TOOLS.caller import (
    apply_bandpass_filter,
    compute_fft,
    remove_dc_offset,
    normalize_signal,
)

fs = 44100
t = np.linspace(0, 1.0, fs, endpoint=False)
sig = np.sin(2*np.pi*440*t) + 0.5*np.sin(2*np.pi*2000*t)

# Preprocess
sig = remove_dc_offset(sig)
sig = normalize_signal(sig)

# Filter (uses assembly if available, else SciPy)
filtered = apply_bandpass_filter(sig, lowcut=300, highcut=3000, fs=fs, order=5)

# FFT helper
freqs, mag = compute_fft(filtered, fs)
```

Run the example entry (customize as needed):

```powershell
python Feature-Conversion/src/main.py
```

## Assembly Acceleration (Optional)

The SOS filtering inner loop is available in x86_64 assembly for Windows/Linux. Build steps (Windows Developer Command Prompt recommended):

```powershell
# Assemble → object
nasm -f win64 Feature-Conversion/lib/TOOLS/src/bandpass_filter.asm `
     -o Feature-Conversion/lib/TOOLS/src/bandpass_filter.obj

# Link → DLL
link /DLL /OUT:Feature-Conversion/lib/TOOLS/src/bandpass_filter.dll `
     Feature-Conversion/lib/TOOLS/src/bandpass_filter.obj msvcrt.lib legacy_stdio_definitions.lib
```

Linux (optional):

```bash
nasm -f elf64 Feature-Conversion/lib/TOOLS/src/bandpass_filter.asm -o Feature-Conversion/lib/TOOLS/src/bandpass_filter.o
gcc -shared -o Feature-Conversion/lib/TOOLS/src/bandpass_filter.so Feature-Conversion/lib/TOOLS/src/bandpass_filter.o -lm
```

Notes:
- The Python wrapper in [Feature-Conversion/lib/TOOLS/caller.py](Feature-Conversion/lib/TOOLS/caller.py) auto-loads the DLL/SO if present and otherwise falls back to SciPy.
- You can also use helper scripts: [Feature-Conversion/lib/TOOLS/build_asm.py](Feature-Conversion/lib/TOOLS/build_asm.py).

## API Reference

All functions below are available from [Feature-Conversion/lib/TOOLS/caller.py](Feature-Conversion/lib/TOOLS/caller.py). Each has a `use_assembly` flag (defaults to True where applicable) and falls back to Python if the assembly library is missing.

- apply_bandpass_filter(data, lowcut, highcut, fs, order=5, use_assembly=True)
- apply_lowpass_filter(data, cutoff, fs, order=5, use_assembly=True)
- apply_highpass_filter(data, cutoff, fs, order=5, use_assembly=True)
- compute_fft(data, fs) -> (freqs, magnitude)
- compute_spectrogram(data, fs, window='hann', nperseg=None)
- compute_welch_psd(data, fs, nperseg=None)
- resample_signal(data, original_rate, target_rate, use_assembly=True)
- remove_dc_offset(data, use_assembly=True)
- normalize_signal(data, use_assembly=True)
- detect_peaks(data, height=None, distance=None, prominence=None)
- smooth_signal(data, window_len=11, window='hanning')
- compute_envelope(data, use_assembly=False)

## Testing & Benchmarking

Quick checks and performance comparison are provided in [Feature-Conversion/lib/TOOLS/test_asm.py](Feature-Conversion/lib/TOOLS/test_asm.py):

```powershell
python Feature-Conversion/lib/TOOLS/test_asm.py
```

This script compares assembly vs Python results (tolerance-based) and prints speedups where applicable.

## Questions

Open an issue or reach out per the contact info in the top-level [DOCUMENTARY.md](DOCUMENTARY.md).

## Academic

If you use this work, please cite the paper (DOI to be added upon publication). A BibTeX entry will be provided here once assigned.

Licensing is per the repository [LICENSE](LICENSE).

## Contributing

Contributions are welcome! Please:
- Keep changes focused and documented.
- Add minimal usage instructions or tests for new features.
- For DSP kernels, prefer clear reference Python first, then optimize.

Thank you for helping improve Feature-Conversion.