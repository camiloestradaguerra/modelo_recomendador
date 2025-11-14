"""
Unit Tests for Data Sampling Component

This test suite ensures the sampling component behaves correctly across various
scenarios, including edge cases and error conditions. Tests follow the AAA pattern
(Arrange-Act-Assert) for clarity and maintainability.

Author: Equipo ADX
Date: 2025-11-13
"""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from main import (
    validate_input_file,
    load_data,
    sample_data,
    save_data,
    run_sampling_pipeline
)


class TestValidateInputFile:
    """Test suite for input file validation logic."""

    def test_valid_parquet_file(self, tmp_path):
        """Should pass validation for existing parquet file."""
        # Arrange
        file_path = tmp_path / "test.parquet"
        df = pd.DataFrame({'a': [1, 2, 3]})
        df.to_parquet(file_path)

        # Act & Assert
        validate_input_file(file_path)  # Should not raise

    def test_nonexistent_file(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        # Arrange
        file_path = tmp_path / "missing.parquet"

        # Act & Assert
        with pytest.raises(FileNotFoundError, match="Input file does not exist"):
            validate_input_file(file_path)

    def test_directory_instead_of_file(self, tmp_path):
        """Should raise ValueError when path points to directory."""
        # Arrange
        dir_path = tmp_path / "test_dir"
        dir_path.mkdir()

        # Act & Assert
        with pytest.raises(ValueError, match="Input path is not a file"):
            validate_input_file(dir_path)

    def test_wrong_file_extension(self, tmp_path):
        """Should raise ValueError for non-parquet file."""
        # Arrange
        file_path = tmp_path / "test.csv"
        file_path.write_text("col1,col2\n1,2\n")

        # Act & Assert
        with pytest.raises(ValueError, match="Input file must be a parquet file"):
            validate_input_file(file_path)


class TestLoadData:
    """Test suite for data loading functionality."""

    def test_load_valid_parquet(self, tmp_path):
        """Should successfully load valid parquet file."""
        # Arrange
        file_path = tmp_path / "data.parquet"
        expected_df = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c']
        })
        expected_df.to_parquet(file_path)

        # Act
        result_df = load_data(file_path)

        # Assert
        pd.testing.assert_frame_equal(result_df, expected_df)

    def test_load_empty_parquet(self, tmp_path):
        """Should handle empty parquet file gracefully."""
        # Arrange
        file_path = tmp_path / "empty.parquet"
        empty_df = pd.DataFrame()
        empty_df.to_parquet(file_path)

        # Act
        result_df = load_data(file_path)

        # Assert
        assert len(result_df) == 0

    def test_load_large_parquet(self, tmp_path):
        """Should handle large parquet files efficiently."""
        # Arrange
        file_path = tmp_path / "large.parquet"
        large_df = pd.DataFrame({
            'id': range(10000),
            'value': range(10000)
        })
        large_df.to_parquet(file_path)

        # Act
        result_df = load_data(file_path)

        # Assert
        assert len(result_df) == 10000
        assert list(result_df.columns) == ['id', 'value']


class TestSampleData:
    """Test suite for data sampling logic."""

    def test_sample_subset(self):
        """Should return correct number of sampled records."""
        # Arrange
        df = pd.DataFrame({'id': range(1000), 'value': range(1000)})
        sample_size = 100

        # Act
        result = sample_data(df, sample_size=sample_size, random_state=42)

        # Assert
        assert len(result) == sample_size
        assert set(result.columns) == {'id', 'value'}

    def test_sample_reproducibility(self):
        """Should produce identical samples with same random state."""
        # Arrange
        df = pd.DataFrame({'id': range(1000), 'value': range(1000)})
        sample_size = 100
        random_state = 42

        # Act
        sample1 = sample_data(df, sample_size, random_state)
        sample2 = sample_data(df, sample_size, random_state)

        # Assert
        pd.testing.assert_frame_equal(sample1, sample2)

    def test_sample_different_with_different_seed(self):
        """Should produce different samples with different random states."""
        # Arrange
        df = pd.DataFrame({'id': range(1000), 'value': range(1000)})
        sample_size = 100

        # Act
        sample1 = sample_data(df, sample_size, random_state=42)
        sample2 = sample_data(df, sample_size, random_state=123)

        # Assert
        assert not sample1['id'].equals(sample2['id'])

    def test_sample_size_exceeds_dataframe(self):
        """Should return all records when sample size exceeds dataframe size."""
        # Arrange
        df = pd.DataFrame({'id': range(50), 'value': range(50)})
        sample_size = 100

        # Act
        result = sample_data(df, sample_size=sample_size)

        # Assert
        assert len(result) == len(df)

    def test_sample_size_equals_dataframe(self):
        """Should return all records when sample size equals dataframe size."""
        # Arrange
        df = pd.DataFrame({'id': range(100), 'value': range(100)})
        sample_size = 100

        # Act
        result = sample_data(df, sample_size=sample_size)

        # Assert
        assert len(result) == len(df)

    def test_sample_preserves_index_reset(self):
        """Should reset index in sampled dataframe."""
        # Arrange
        df = pd.DataFrame({'id': range(100), 'value': range(100)})
        df.index = range(50, 150)  # Non-sequential index
        sample_size = 20

        # Act
        result = sample_data(df, sample_size=sample_size, random_state=42)

        # Assert
        assert result.index.tolist() == list(range(sample_size))


class TestSaveData:
    """Test suite for data saving functionality."""

    def test_save_dataframe(self, tmp_path):
        """Should successfully save dataframe to parquet."""
        # Arrange
        df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
        output_path = tmp_path / "output.parquet"

        # Act
        save_data(df, output_path)

        # Assert
        assert output_path.exists()
        loaded_df = pd.read_parquet(output_path)
        pd.testing.assert_frame_equal(df, loaded_df)

    def test_save_creates_parent_directories(self, tmp_path):
        """Should create parent directories if they don't exist."""
        # Arrange
        df = pd.DataFrame({'a': [1, 2, 3]})
        output_path = tmp_path / "nested" / "dir" / "output.parquet"

        # Act
        save_data(df, output_path)

        # Assert
        assert output_path.exists()
        assert output_path.parent.exists()

    def test_save_empty_dataframe(self, tmp_path):
        """Should handle saving empty dataframe."""
        # Arrange
        df = pd.DataFrame()
        output_path = tmp_path / "empty.parquet"

        # Act
        save_data(df, output_path)

        # Assert
        assert output_path.exists()
        loaded_df = pd.read_parquet(output_path)
        assert len(loaded_df) == 0


class TestRunSamplingPipeline:
    """Integration tests for the complete sampling pipeline."""

    def test_full_pipeline_success(self, tmp_path):
        """Should execute complete pipeline successfully."""
        # Arrange
        input_path = tmp_path / "input.parquet"
        output_path = tmp_path / "output.parquet"

        # Create input data
        df = pd.DataFrame({
            'id': range(500),
            'value': range(500),
            'category': ['A', 'B', 'C'] * 166 + ['A', 'B']
        })
        df.to_parquet(input_path)

        # Act
        run_sampling_pipeline(
            input_path=str(input_path),
            output_path=str(output_path),
            sample_size=100,
            random_state=42
        )

        # Assert
        assert output_path.exists()
        result_df = pd.read_parquet(output_path)
        assert len(result_df) == 100
        assert set(result_df.columns) == {'id', 'value', 'category'}

    def test_pipeline_with_nested_output(self, tmp_path):
        """Should create nested directories for output."""
        # Arrange
        input_path = tmp_path / "input.parquet"
        output_path = tmp_path / "nested" / "output" / "sample.parquet"

        df = pd.DataFrame({'a': range(100)})
        df.to_parquet(input_path)

        # Act
        run_sampling_pipeline(
            input_path=str(input_path),
            output_path=str(output_path),
            sample_size=50,
            random_state=42
        )

        # Assert
        assert output_path.exists()
        result_df = pd.read_parquet(output_path)
        assert len(result_df) == 50

    def test_pipeline_reproducibility(self, tmp_path):
        """Should produce identical results with same parameters."""
        # Arrange
        input_path = tmp_path / "input.parquet"
        output_path1 = tmp_path / "output1.parquet"
        output_path2 = tmp_path / "output2.parquet"

        df = pd.DataFrame({'id': range(1000), 'value': range(1000)})
        df.to_parquet(input_path)

        # Act
        run_sampling_pipeline(
            input_path=str(input_path),
            output_path=str(output_path1),
            sample_size=100,
            random_state=42
        )
        run_sampling_pipeline(
            input_path=str(input_path),
            output_path=str(output_path2),
            sample_size=100,
            random_state=42
        )

        # Assert
        df1 = pd.read_parquet(output_path1)
        df2 = pd.read_parquet(output_path2)
        pd.testing.assert_frame_equal(df1, df2)


@pytest.fixture
def sample_dataframe():
    """Fixture providing a sample dataframe for tests."""
    return pd.DataFrame({
        'id': range(100),
        'value': range(100, 200),
        'category': ['A', 'B', 'C', 'D'] * 25
    })


@pytest.fixture
def temp_parquet_file(tmp_path, sample_dataframe):
    """Fixture providing a temporary parquet file."""
    file_path = tmp_path / "test.parquet"
    sample_dataframe.to_parquet(file_path)
    return file_path
