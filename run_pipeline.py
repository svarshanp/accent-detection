"""
Master Pipeline - Run all steps end-to-end.
Includes error handling so individual step failures don't crash the whole pipeline.
"""
import os
import sys
import time
import io
import traceback

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_step(step_num, total_steps, step_name, func, *args, **kwargs):
    """
    Run a single pipeline step with error handling.
    
    Returns:
        True if step succeeded, False if it failed.
    """
    print("\n\n" + "=" * 60)
    print(f"  STEP {step_num}/{total_steps}: {step_name}")
    print("=" * 60)
    
    step_start = time.time()
    try:
        func(*args, **kwargs)
        elapsed = time.time() - step_start
        print(f"\n  ✓ {step_name} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - step_start
        print(f"\n  ✗ {step_name} FAILED after {elapsed:.1f}s")
        print(f"  Error: {e}")
        print(f"  Traceback:")
        traceback.print_exc()
        return False


def main():
    total_start = time.time()
    total_steps = 6
    results = {}
    
    print("\n" + "#" * 60)
    print("#  ACCENT DETECTION - Full Pipeline Execution")
    print("#" * 60)
    
    # Step 1: Generate dataset
    from src.data_loader import generate_dataset
    results['Dataset Generation'] = run_step(
        1, total_steps, "Generating Dataset",
        generate_dataset, samples_per_class=500
    )
    
    # Step 2: Extract features (depends on Step 1)
    if results['Dataset Generation']:
        from src.feature_extractor import extract_all_features
        results['Feature Extraction'] = run_step(
            2, total_steps, "Extracting Features",
            extract_all_features
        )
    else:
        print(f"\n  ⚠ Skipping Step 2 (Feature Extraction) — dataset generation failed")
        results['Feature Extraction'] = False
    
    # Step 3: Train ML models (depends on Step 2)
    if results['Feature Extraction']:
        from src.train_ml import train_ml_models
        results['ML Training'] = run_step(
            3, total_steps, "Training ML Models",
            train_ml_models
        )
    else:
        print(f"\n  ⚠ Skipping Step 3 (ML Training) — feature extraction failed")
        results['ML Training'] = False
    
    # Step 4: Train DL models (depends on Step 2)
    if results['Feature Extraction']:
        from src.train_dl import train_dl_models
        results['DL Training'] = run_step(
            4, total_steps, "Training DL Models",
            train_dl_models
        )
    else:
        print(f"\n  ⚠ Skipping Step 4 (DL Training) — feature extraction failed")
        results['DL Training'] = False
    
    # Step 5: Train Transformer model (depends on Step 2)
    if results['Feature Extraction']:
        from src.train_transformer import train_transformer
        results['Transformer Training'] = run_step(
            5, total_steps, "Training Transformer Model",
            train_transformer
        )
    else:
        print(f"\n  ⚠ Skipping Step 5 (Transformer Training) — feature extraction failed")
        results['Transformer Training'] = False
    
    # Step 6: Evaluate (runs if any model was trained)
    any_model_trained = any([
        results.get('ML Training', False),
        results.get('DL Training', False),
        results.get('Transformer Training', False),
    ])
    if any_model_trained:
        from src.evaluate import run_evaluation
        results['Evaluation'] = run_step(
            6, total_steps, "Generating Evaluation Plots",
            run_evaluation
        )
    else:
        print(f"\n  ⚠ Skipping Step 6 (Evaluation) — no models were trained")
        results['Evaluation'] = False
    
    # Summary
    elapsed = time.time() - total_start
    succeeded = sum(1 for v in results.values() if v)
    failed = sum(1 for v in results.values() if not v)
    
    print("\n\n" + "#" * 60)
    print(f"#  PIPELINE COMPLETE in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("#" * 60)
    
    print(f"\n  Results: {succeeded} succeeded, {failed} failed")
    print()
    for step_name, success in results.items():
        status = "✓" if success else "✗"
        print(f"    {status}  {step_name}")
    
    if failed > 0:
        print(f"\n  ⚠ Some steps failed. Check the errors above for details.")
    else:
        print(f"\n  All steps completed successfully!")
    
    print(f"\n  Next: Run the app with:")
    print(f"  streamlit run app/app.py")
    print()
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
