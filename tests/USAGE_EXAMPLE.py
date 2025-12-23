"""
Usage Example for Feature Extraction Module

This demonstrates how to use the refactored feature extraction system.
"""
from pathlib import Path
import sys

# Add lib to path
sys.path.insert(0, str(Path(__file__).parent / 'lib'))

from mod import Features, Extraction, Conversion, extract_all_tiers, RESOURCE_DIR, RESULT_DIR


def example_basic_usage():
    """Basic usage: Extract features from default directory"""
    print("="*60)
    print("EXAMPLE 1: Basic Feature Extraction")
    print("="*60)
    
    # Option 1: Use Features for raw processing (if needed)
    features = Features(source=str(RESOURCE_DIR))
    print(f"Source directory: {features.source}")
    print(f"Audio files found: {len(features._list_audio_files())}")
    
    # Option 2: Use Extraction with processed data
    # Pass Features instance or directory path as data
    extraction = Extraction(data=features)
    
    # Run tier 1 (basic features)
    print("\n--- Running Tier 1 ---")
    tier1_results = extraction.tier_one()
    print(f"Tier 1 extracted {len(tier1_results)} files")
    
    # Run tier 2 (advanced features)
    print("\n--- Running Tier 2 ---")
    tier2_results = extraction.tier_two()
    print(f"Tier 2 extracted {len(tier2_results)} files")
    
    # Run tier 3 (DNN embeddings)
    print("\n--- Running Tier 3 ---")
    tier3_results = extraction.tier_three()
    print(f"Tier 3 extracted {len(tier3_results)} files")


def example_with_custom_path():
    """Extract features from a custom directory"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Custom Directory")
    print("="*60)
    
    custom_path = "./Data/Folders"  # Change this to your audio directory
    
    # Direct initialization with path
    extraction = Extraction(data=custom_path)
    
    # Run all tiers
    tier1 = extraction.tier_one()
    tier2 = extraction.tier_two()
    tier3 = extraction.tier_three()
    
    print(f"\nResults saved to: {RESULT_DIR}")


def example_convenience_function():
    """Use the convenience function to extract all tiers at once"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Convenience Function")
    print("="*60)
    
    # Extract all tiers at once
    all_results = extract_all_tiers(source=str(RESOURCE_DIR))
    
    print("\nAll tiers completed:")
    print(f"  Tier 1: {len(all_results['tier1'])} files")
    print(f"  Tier 2: {len(all_results['tier2'])} files")
    print(f"  Tier 3: {len(all_results['tier3'])} files")


def example_conversion():
    """Load and convert features for model training"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Feature Conversion")
    print("="*60)
    
    # Load extracted features
    conversion = Conversion(data={})
    
    # Load features from files
    tier1_features = conversion.load_features(tier='tier1')
    tier2_features = conversion.load_features(tier='tier2')
    tier3_features = conversion.load_features(tier='tier3')
    
    print(f"Loaded features:")
    print(f"  Tier 1: {len(tier1_features)} files")
    print(f"  Tier 2: {len(tier2_features)} files")
    print(f"  Tier 3: {len(tier3_features)} files")
    
    # Convert to desired format
    # converted = conversion.convert(output_format='numpy')
    print("\nFeatures ready for model training!")


def example_inspect_features():
    """Inspect the contents of saved features"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Inspect Feature Contents")
    print("="*60)
    
    import pickle
    from pathlib import Path
    
    # Find first tier1 file
    tier1_files = list(RESULT_DIR.glob("*_tier1.pkl"))
    
    if tier1_files:
        with open(tier1_files[0], 'rb') as f:
            features = pickle.load(f)
        
        print(f"\nFile: {tier1_files[0].name}")
        print("\nFeature shapes:")
        for key, value in features.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")
            else:
                print(f"  {key}: {value}")
    else:
        print("No feature files found. Run extraction first!")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Audio Feature Extraction - Usage Examples")
    print("="*60)
    
    # Check if audio files exist
    if not RESOURCE_DIR.exists():
        print(f"\nWarning: Resource directory not found: {RESOURCE_DIR}")
        print("Creating directory...")
        RESOURCE_DIR.mkdir(parents=True, exist_ok=True)
    
    audio_files = list(RESOURCE_DIR.glob("*.wav")) + \
                  list(RESOURCE_DIR.glob("*.mp3")) + \
                  list(RESOURCE_DIR.glob("*.flac"))
    
    if not audio_files:
        print(f"\nNo audio files found in {RESOURCE_DIR}")
        print("Please add some audio files (.wav, .mp3, .flac) and run again.")
        print("\nExample: Copy audio files to:")
        print(f"  {RESOURCE_DIR}")
    else:
        print(f"\nFound {len(audio_files)} audio files in {RESOURCE_DIR}")
        
        # Run examples
        try:
            # Uncomment the examples you want to run:
            
            example_basic_usage()
            # example_with_custom_path()
            # example_convenience_function()
            # example_conversion()
            # example_inspect_features()
            
        except ImportError as e:
            print(f"\n\nError: {e}")
            print("\nPlease install required packages:")
            print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()
