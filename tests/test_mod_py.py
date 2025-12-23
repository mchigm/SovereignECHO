import os
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Ensure the lib directory is on path
LIB_DIR = Path(__file__).resolve().parents[1] / "Feature-Conversion" / "lib"
sys.path.insert(0, str(LIB_DIR))

import mod


class FakeTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=float)
        self.device = 'cpu'

    def __array__(self):
        return self.array

    @property
    def shape(self):
        return self.array.shape

    def detach(self):
        return self

    def cpu(self):
        return self

    def to(self, device):
        return self

    def squeeze(self, axis=None):
        return FakeTensor(np.squeeze(self.array, axis=axis))

    def view(self, *shape):
        return FakeTensor(self.array.reshape(shape))

    def sum(self, dim=None):
        axis = dim if dim is not None else None
        return FakeTensor(self.array.sum(axis=axis))

    def transpose(self, *axes):
        return FakeTensor(self.array.transpose(*axes))

    def __getitem__(self, item):
        return FakeTensor(self.array[item])

    def __setitem__(self, key, value):
        self.array[key] = np.asarray(value, dtype=float)

    def unsqueeze(self, axis):
        return FakeTensor(np.expand_dims(self.array, axis))

    def abs(self):
        return FakeTensor(np.abs(self.array))

    def __add__(self, other):
        other_arr = other.array if hasattr(other, 'array') else other
        return FakeTensor(self.array + other_arr)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        other_arr = other.array if hasattr(other, 'array') else other
        return FakeTensor(self.array * other_arr)

    def __rmul__(self, other):
        return self.__mul__(other)

    def reshape(self, *shape):
        return FakeTensor(self.array.reshape(shape))


def make_fake_torch_module():
    fake = types.ModuleType("torch")

    class FakeCuda:
        @staticmethod
        def is_available():
            return False

        @staticmethod
        def empty_cache():
            return None

    fake.cuda = FakeCuda()
    fake.device = lambda arg: f"device({arg})"

    def log(x):
        arr = x.array if hasattr(x, 'array') else np.asarray(x)
        return FakeTensor(np.log(arr + 1e-10))

    def arange(n, device=None):
        return FakeTensor(np.arange(n, dtype=float))

    def cos(x):
        arr = x.array if hasattr(x, 'array') else np.asarray(x)
        return FakeTensor(np.cos(arr))

    def matmul(a, b):
        arr_a = a.array if hasattr(a, 'array') else np.asarray(a)
        arr_b = b.array if hasattr(b, 'array') else np.asarray(b)
        return FakeTensor(np.matmul(arr_a, arr_b))

    fake.log = log
    fake.arange = arange
    fake.cos = cos
    fake.matmul = matmul

    class NoGrad:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    fake.no_grad = lambda: NoGrad()

    class FakeModule:
        def __call__(self, x):
            return self.forward(x)

        def eval(self):
            return None

        def to(self, device):
            return self

    class IdentityLayer:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

    class FakeFlatten:
        def __call__(self, x):
            arr = x.array if hasattr(x, 'array') else np.asarray(x)
            batch = arr.shape[0]
            return FakeTensor(arr.reshape(batch, -1))

    class FakeLinear:
        def __init__(self, in_features, out_features):
            self.out_features = out_features

        def __call__(self, x):
            batch = x.shape[0]
            return FakeTensor(np.zeros((batch, self.out_features)))

    class FakeSequential:
        def __init__(self, *layers):
            self.layers = layers

        def __call__(self, x):
            result = x
            for layer in self.layers:
                result = layer(result)
            return result

    nn = types.SimpleNamespace(
        Module=FakeModule,
        Sequential=FakeSequential,
        Conv2d=IdentityLayer,
        ReLU=IdentityLayer,
        AdaptiveAvgPool2d=IdentityLayer,
        Flatten=FakeFlatten,
        Linear=FakeLinear,
    )

    fake.nn = nn
    return fake


@pytest.fixture(autouse=True)
def isolate_paths(tmp_path, monkeypatch):
    resource = tmp_path / "Data" / "Folders"
    result = tmp_path / "Data" / "RESULT"
    resource.mkdir(parents=True)
    result.mkdir(parents=True)
    monkeypatch.setattr(mod, "RESOURCE_DIR", resource)
    monkeypatch.setattr(mod, "RESULT_DIR", result)
    yield resource, result


@pytest.fixture
def fake_torch(monkeypatch):
    torch_mod = make_fake_torch_module()
    monkeypatch.setitem(sys.modules, 'torch', torch_mod)
    yield torch_mod


def patch_tier_one_methods(extraction):
    zero_audio = np.zeros((1, 16000))

    extraction._load_audio = lambda path, target_sr=16000: (zero_audio, target_sr)
    extraction._stft_mag = lambda wav, n_fft, hop_length: np.ones((1, 1, 10))
    extraction._spectral_centroid = lambda mag, sr: np.ones(10)
    extraction._zcr = lambda wav, frame_length, hop_length: np.ones(10)
    extraction._energy = lambda wav, frame_length, hop_length: np.ones(10)
    extraction._mel_spectrogram = lambda sr, n_mels, n_fft, hop_length: lambda wav: np.ones((1, 128, 8))
    extraction._mfcc = lambda sr, n_mfcc, n_mels, n_fft, hop_length: lambda wav: np.ones((1, 13, 8))
    extraction._safe_to_numpy = lambda t: t


def patch_tier_two_methods(extraction):
    extraction._load_audio = lambda path, target_sr=16000: (FakeTensor(np.ones((1, 1, 16))), target_sr)
    def fake_cqt(sr, bins_per_octave, n_bins, hop_length):
        del sr, bins_per_octave, n_bins, hop_length
        return lambda wav: FakeTensor(np.ones((1, 36, 20)))
    extraction._cqt = fake_cqt
    extraction._mel_spectrogram = lambda sr, n_mels, n_fft, hop_length: lambda wav: FakeTensor(np.ones((1, 64, 8)))
    extraction._safe_to_numpy = lambda t: t.array if hasattr(t, 'array') else np.asarray(t)


def patch_tier_three_methods(extraction):
    extraction._load_audio = lambda path, target_sr=16000: (FakeTensor(np.ones((1, 1, 16))), target_sr)
    extraction._mel_spectrogram = lambda sr, n_mels, n_fft, hop_length: lambda wav: FakeTensor(np.ones((1, 128, 8)))
    extraction._safe_to_numpy = lambda t: t.array if hasattr(t, 'array') else np.asarray(t)


def test_validate_audio_file(tmp_path):
    audio_file = tmp_path / "signal.wav"
    audio_file.write_bytes(b"dummy")
    assert mod.validate_audio_file(str(audio_file))


def test_features_list_audio_files(monkeypatch, tmp_path):
    resource = tmp_path / "Data" / "Folders"
    resource.mkdir(parents=True, exist_ok=True)
    (resource / "voice.mp3").write_bytes(b"data")
    monkeypatch.setattr(mod, "RESOURCE_DIR", resource)
    features = mod.Features()
    files = features._list_audio_files()
    assert len(files) == 1


def test_gpu_context_detects_device(monkeypatch):
    fake = make_fake_torch_module()
    monkeypatch.setitem(sys.modules, 'torch', fake)
    ctx = mod.GPUContext()
    assert ctx.device == "device(cpu)"
    assert not ctx.has_gpu


def test_tier_one_creates_features(isolate_paths, fake_torch):
    resource, result = isolate_paths
    (resource / "voice.wav").write_bytes(b"audio")
    extraction = mod.Extraction(data=str(resource))
    patch_tier_one_methods(extraction)
    features = extraction.tier_one()
    assert "voice" in features
    assert (result / "voice_tier1.pkl").exists()


def test_tier_two_handles_empty(isolate_paths, monkeypatch):
    resource, _ = isolate_paths
    monkeypatch.setattr(mod.Extraction, "_list_audio_files", lambda self: [])
    extraction = mod.Extraction(data=str(resource))
    assert extraction.tier_two() == {}


def test_tier_three_handles_empty(isolate_paths, monkeypatch):
    resource, _ = isolate_paths
    monkeypatch.setattr(mod.Extraction, "_list_audio_files", lambda self: [])
    extraction = mod.Extraction(data=str(resource))
    assert extraction.tier_three() == {}


def test_conversion_load_and_convert(monkeypatch, isolate_paths):
    resource, result = isolate_paths
    data = {"value": 1}
    sample = result / "test_tier1.pkl"
    with open(sample, 'wb') as f:
        pickle.dump(data, f)
    conversion = mod.Conversion(data={})
    loaded = conversion.load_features('tier1')
    assert "test_tier1" in loaded
    assert loaded["test_tier1"] == data
    assert conversion.convert('numpy') == {}
    fake = types.ModuleType('torch')
    monkeypatch.setitem(sys.modules, 'torch', fake)
    assert conversion.convert('torch') == {}
    fake_pd = types.ModuleType('pandas')
    fake_pd.DataFrame = lambda: "df"
    monkeypatch.setitem(sys.modules, 'pandas', fake_pd)
    assert conversion.convert('pandas') == "df"
    with pytest.raises(ValueError):
        conversion.convert('something-unknown')


def test_extract_all_tiers(monkeypatch):
    monkeypatch.setattr(mod.Extraction, 'tier_one', lambda self: {'tier1': {}})
    monkeypatch.setattr(mod.Extraction, 'tier_two', lambda self: {'tier2': {}})
    monkeypatch.setattr(mod.Extraction, 'tier_three', lambda self: {'tier3': {}})
    results = mod.extract_all_tiers(source=str(mod.RESOURCE_DIR))
    assert 'tier1' in results
    assert 'tier2' in results
    assert 'tier3' in results


def test_architecture_classes_instantiation():
    arch = mod.Architecture()
    train = mod.Training()
    fine = mod.FineTuning()
    mod_obj = mod.Modification()
    build = mod.Building()
    res = mod.Result()
    assert arch.description == "Model architecture design"
    assert train.description == "Model training pipeline"
    assert fine.description == "Model fine-tuning"
    assert mod_obj.description == "Model modification"
    assert build.description == "Model building"
    assert res.description == "Result analysis"
