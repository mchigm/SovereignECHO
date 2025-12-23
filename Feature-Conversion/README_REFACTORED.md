# Feature Extraction Module - Refactored

## Overview

This module provides GPU-accelerated audio feature extraction for human vocal analysis with three tiers of processing:

- **Tier 1**: Basic spectral and temporal features (STFT, MFCC, ZCR, energy, spectral centroid)
- **Tier 2**: Advanced ML-ready features (CQCC, chroma, Mel-DCT)
- **Tier 3**: Deep neural network embeddings

## Key Improvements

### 1. Fixed Data Flow Architecture

**Before**: `Extraction` inherited from `Features` and confused raw audio paths with processed data.

**After**: Clear separation:
- `Features`: Base class for raw audio file handling
- `Extraction`: Takes processed data/Features instance, performs GPU-accelerated transforms
- `Conversion`: Handles feature format conversion for model training

### 2. GPU Memory Management

- Added `GPUContext` manager for automatic CUDA cache clearing
- Prevents OOM errors during batch processing
- Automatic fallback to CPU if GPU unavailable

### 3. Better Error Handling

- Validation for audio files
- Progress tracking with file counts
- Graceful error recovery (continues on failure)
- Clear success/failure indicators

### 4. Type Hints & Documentation

- Full type annotations
- Comprehensive docstrings
- Clear return types for all methods

### 5. Improved Code Organization

- Constants at module level (`ROOT_PATH`, `RESOURCE_DIR`, `RESULT_DIR`)
- Utility functions (`inquiry()`, `validate_audio_file()`)
- Consistent naming conventions
- Removed unused imports

## Installation

### Required Dependencies

```powershell
# Install PyTorch with CUDA 12.1 (for GPU acceleration)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or CPU-only version
pip install torch torchaudio

# Optional for data conversion
pip install numpy pandas
```

## Usage

### Basic Usage

```python
from lib.mod import Extraction, Features

# Option 1: Pass Features instance
features = Features(source="./Data/Folders")
extraction = Extraction(data=features)

# Option 2: Pass directory path directly
extraction = Extraction(data="./Data/Folders")

# Run extraction tiers
tier1_results = extraction.tier_one()
tier2_results = extraction.tier_two()
tier3_results = extraction.tier_three()
```

### Convenience Function

```python
from lib.mod import extract_all_tiers

# Extract all tiers at once
results = extract_all_tiers(source="./Data/Folders")

# Access results
print(f"Tier 1: {len(results['tier1'])} files")
print(f"Tier 2: {len(results['tier2'])} files")
print(f"Tier 3: {len(results['tier3'])} files")
```

### Feature Conversion

```python
from lib.mod import Conversion

conversion = Conversion(data={})

# Load saved features
tier1_features = conversion.load_features(tier='tier1')
tier2_features = conversion.load_features(tier='tier2')
tier3_features = conversion.load_features(tier='tier3')

# Convert to desired format
# numpy_data = conversion.convert(output_format='numpy')
# torch_data = conversion.convert(output_format='torch')
# df = conversion.convert(output_format='pandas')
```

### Loading Saved Features

```python
import pickle
from pathlib import Path

# Features are saved as pickle files in Data/RESULT/
result_dir = Path("Data/RESULT")

# Load a specific file
with open(result_dir / "audio_file_tier1.pkl", 'rb') as f:
    features = pickle.load(f)

# Access individual features
print(features['mfcc'].shape)           # MFCC coefficients
print(features['mel_spectrogram'].shape) # Mel spectrogram
print(features['energy'].shape)          # Frame energy
```

## Feature Descriptions

### Tier 1: Basic Features

| Feature | Shape | Description |
|---------|-------|-------------|
| `stft_mag` | `[freq, time]` | STFT magnitude spectrum |
| `spectral_centroid` | `[time]` | Center of mass of spectrum |
| `zcr` | `[frames]` | Zero-crossing rate per frame |
| `energy` | `[frames]` | Average energy per frame |
| `mel_spectrogram` | `[n_mels, time]` | Mel-scaled spectrogram |
| `mfcc` | `[n_mfcc, time]` | Mel-frequency cepstral coefficients |

### Tier 2: Advanced Features

| Feature | Shape | Description |
|---------|-------|-------------|
| `cqcc` | `[n_bins, time]` | Constant-Q cepstral coefficients |
| `chroma` | `[12, time]` | Pitch class profiles |
| `mel_dct` | `[n_mels, time]` | DCT of log-mel features |

### Tier 3: Deep Features

| Feature | Shape | Description |
|---------|-------|-------------|
| `dnn_embedding` | `[128]` | High-level CNN embedding |

## File Structure

```
Feature-Conversion/
├── lib/
│   ├── mod.pyw              # Main refactored module
│   ├── LIB/
│   │   ├── tools.py
│   │   └── __init__.py
│   └── TOOLS/
│       ├── asm_interface.py
│       ├── build_asm.py
│       ├── caller.py
│       └── ...
├── Data/
│   ├── Folders/             # Input: Place audio files here
│   └── RESULT/              # Output: Extracted features saved here
├── USAGE_EXAMPLE.py         # Usage examples
└── README_REFACTORED.md     # This file
```

## Output Format

Features are saved as pickle files:
- `filename_tier1.pkl` - Tier 1 features
- `filename_tier2.pkl` - Tier 2 features
- `filename_tier3.pkl` - Tier 3 features

Each file contains a dictionary with feature arrays as NumPy arrays.

## GPU Acceleration

The module automatically detects and uses CUDA GPUs when available:

```python
extraction = Extraction(data="./Data/Folders")
# Prints: "Extraction initialized | Device: cuda:0 | GPU: True"
```

All heavy computations (STFT, Mel transforms, CQT, CNN) run on GPU for ~10-100x speedup.

## Performance Tips

1. **Batch Processing**: Process multiple files at once for better GPU utilization
2. **Memory**: ~2-4GB GPU memory per file (adjust batch size if OOM)
3. **Sample Rate**: Default 16kHz balances quality and speed
4. **Transform Caching**: Transforms are reused across files for efficiency

## Troubleshooting

### "Import torchaudio could not be resolved"
Install: `pip install torch torchaudio`

### "CUDA out of memory"
Reduce batch size or process files sequentially

### "No audio files found"
Check files are in `Data/Folders/` with extensions: `.wav`, `.mp3`, `.flac`, `.ogg`, `.m4a`

### Features have NaN values
Check audio file integrity and sample rate compatibility

## Future Enhancements

- [ ] Pretrained vocal embedding models (Wav2Vec2, HuBERT)
- [ ] Multi-GPU support
- [ ] Real-time streaming feature extraction
- [ ] Feature normalization and standardization
- [ ] Augmentation pipeline integration
- [ ] Model training integration in `Training` class

## Credits

Refactored December 2025 for improved clarity, GPU efficiency, and maintainability.
