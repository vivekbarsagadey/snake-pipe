"""
Tests for the main pipeline functionality
"""

import unittest
import pandas as pd
from unittest.mock import patch, MagicMock
from snake_pipe.pipeline import Pipeline


class TestPipeline(unittest.TestCase):
    """Test cases for the Pipeline class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.pipeline = Pipeline("test_pipeline")
        
        # Sample test data
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 30, 35, 40, 45],
            'email': ['alice@test.com', 'bob@test.com', 'charlie@test.com', 
                     'david@test.com', 'eve@test.com']
        })
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization"""
        self.assertEqual(self.pipeline.name, "test_pipeline")
        self.assertEqual(len(self.pipeline.steps), 0)
        self.assertIsNone(self.pipeline.data)
        self.assertEqual(self.pipeline.metadata, {})
    
    @patch('snake_pipe.extract.CSVExtractor')
    def test_extract_from_csv(self, mock_extractor):
        """Test CSV extraction step"""
        # Mock the extractor
        mock_instance = MagicMock()
        mock_instance.extract.return_value = self.sample_data
        mock_extractor.return_value = mock_instance
        
        # Add extraction step
        result = self.pipeline.extract_from_csv("test.csv")
        
        # Verify method chaining
        self.assertIs(result, self.pipeline)
        self.assertEqual(len(self.pipeline.steps), 1)
        self.assertEqual(self.pipeline.steps[0][0], 'extract_csv')
    
    def test_apply_custom_transform(self):
        """Test custom transformation step"""
        def double_age(df):
            df_copy = df.copy()
            df_copy['age'] = df_copy['age'] * 2
            return df_copy
        
        # Add custom transform step
        result = self.pipeline.apply_custom_transform(double_age, "double_age")
        
        # Verify method chaining
        self.assertIs(result, self.pipeline)
        self.assertEqual(len(self.pipeline.steps), 1)
        self.assertEqual(self.pipeline.steps[0][0], 'custom_transform_double_age')
    
    def test_get_metadata(self):
        """Test metadata retrieval"""
        # Add some steps
        self.pipeline.extract_from_csv("test.csv")
        self.pipeline.clean_data()
        
        metadata = self.pipeline.get_metadata()
        
        self.assertEqual(metadata['pipeline_name'], "test_pipeline")
        self.assertEqual(metadata['steps_count'], 2)
        self.assertEqual(len(metadata['steps']), 2)
        self.assertIn('extract_csv', metadata['steps'])
        self.assertIn('clean_data', metadata['steps'])
    
    def test_preview_no_data(self):
        """Test preview when no data is available"""
        result = self.pipeline.preview()
        self.assertIsNone(result)
    
    def test_preview_with_data(self):
        """Test preview when data is available"""
        self.pipeline.data = self.sample_data
        
        result = self.pipeline.preview(3)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        self.assertEqual(list(result.columns), ['id', 'name', 'age', 'email'])


class TestPipelineExecution(unittest.TestCase):
    """Test cases for pipeline execution"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_data = pd.DataFrame({
            'id': [1, 2, 3],
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35]
        })
    
    @patch('snake_pipe.extract.CSVExtractor')
    @patch('snake_pipe.load.FileLoader')
    def test_simple_pipeline_execution(self, mock_file_loader, mock_csv_extractor):
        """Test execution of a simple pipeline"""
        # Mock extractor
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract.return_value = self.sample_data
        mock_csv_extractor.return_value = mock_extractor_instance
        
        # Mock loader
        mock_loader_instance = MagicMock()
        mock_loader_instance.load_csv.return_value = "/path/to/output.csv"
        mock_file_loader.return_value = mock_loader_instance
        
        # Create and run pipeline
        pipeline = Pipeline("test_execution")
        result = (pipeline
                 .extract_from_csv("input.csv")
                 .load_to_csv("output.csv")
                 .run())
        
        # Verify execution
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 3)
        mock_extractor_instance.extract.assert_called_once()
        mock_loader_instance.load_csv.assert_called_once()
    
    def test_pipeline_execution_no_data(self):
        """Test pipeline execution without data extraction"""
        pipeline = Pipeline("test_no_data")
        
        with self.assertRaises(ValueError):
            pipeline.clean_data().run()
    
    @patch('snake_pipe.extract.CSVExtractor')
    def test_custom_transform_execution(self, mock_csv_extractor):
        """Test execution with custom transformation"""
        # Mock extractor
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.extract.return_value = self.sample_data
        mock_csv_extractor.return_value = mock_extractor_instance
        
        def add_full_name(df):
            df_copy = df.copy()
            df_copy['full_name'] = df_copy['name'] + " (Age: " + df_copy['age'].astype(str) + ")"
            return df_copy
        
        # Create and run pipeline
        pipeline = Pipeline("test_custom_transform")
        result = (pipeline
                 .extract_from_csv("input.csv")
                 .apply_custom_transform(add_full_name, "add_full_name")
                 .run())
        
        # Verify transformation
        self.assertIn('full_name', result.columns)
        self.assertEqual(result['full_name'].iloc[0], "Alice (Age: 25)")


if __name__ == '__main__':
    unittest.main()