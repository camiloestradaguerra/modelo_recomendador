"""Quick pipeline tests without full training"""

import sys
from pathlib import Path

def test_step1_sampling():
    """Test that sampling completed successfully."""
    output_file = Path("data/02-sampled/sampled_data.parquet")
    assert output_file.exists(), "Sampling output not found"
    
    import pandas as pd
    df = pd.read_parquet(output_file)
    assert len(df) == 2000, f"Expected 2000 records, got {len(df)}"
    print(f"PASS: Sampling - {len(df)} records")

def test_step2_features():
    """Test that feature engineering completed successfully."""
    output_file = Path("data/03-features/features.parquet")
    encoders_file = Path("models/label_encoders.pkl")
    
    assert output_file.exists(), "Features output not found"
    assert encoders_file.exists(), "Encoders not found"
    
    import pandas as pd
    df = pd.read_parquet(output_file)
    assert df.shape[1] > 40, f"Expected 40+ features, got {df.shape[1]}"
    print(f"PASS: Features - {df.shape[1]} features, {len(df)} records")

def test_github_actions_syntax():
    """Test that GitHub Actions workflow is valid YAML."""
    workflow_file = Path(".github/workflows/ml-pipeline.yml")
    assert workflow_file.exists(), "GitHub Actions workflow not found"
    
    import yaml
    with open(workflow_file, 'r') as f:
        config = yaml.safe_load(f)
    
    assert 'jobs' in config, "No jobs defined in workflow"
    assert 'test' in config['jobs'], "No test job found"
    assert 'pipeline' in config['jobs'], "No pipeline job found"
    assert 'cleanup' in config['jobs'], "No cleanup job found"
    
    print(f"PASS: GitHub Actions - {len(config['jobs'])} jobs defined")

def run_quick_tests():
    """Run quick pipeline tests."""
    print("Running quick pipeline tests...")
    print("=" * 50)
    
    try:
        test_step1_sampling()
        test_step2_features()
        test_github_actions_syntax()
        
        print("=" * 50)
        print("All quick tests passed!")
        return True
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        return False
    except Exception as e:
        print(f"\nError: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_tests()
    sys.exit(0 if success else 1)
