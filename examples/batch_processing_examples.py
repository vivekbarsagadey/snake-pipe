"""Example usage and demonstration of the batch processing engine.

This module provides practical examples of how to use the batch processing
engine for different scenarios and configurations.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List

from snake_pipe.extract.batch_config import BatchConfig, BatchStrategy, ErrorHandlingMode
from snake_pipe.extract.batch_processor import BatchProcessingEngine
from snake_pipe.extract.models import ASTFile
from snake_pipe.utils.checkpoint_manager import CheckpointManager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_processor():
    """Create a sample processor function for demonstration.
    
    In a real implementation, this would connect to your actual
    AST processing logic.
    """
    def process_ast_files(files: List[ASTFile]) -> List[ASTFile]:
        """Sample processor that simulates AST processing.
        
        Args:
            files: List of AST files to process
            
        Returns:
            List[ASTFile]: Processed files
        """
        processed_files = []
        
        for file in files:
            try:
                # Simulate processing time
                import time
                time.sleep(0.01)  # 10ms per file
                
                # In real implementation, you would:
                # 1. Parse the AST file
                # 2. Validate the content
                # 3. Transform/normalize the data
                # 4. Add metadata
                # 5. Write to output location
                
                # For demo, just mark as processed
                file.metadata.processed_time = datetime.now()
                processed_files.append(file)
                
            except Exception as e:
                logger.error(f"Failed to process file {file.path}: {e}")
                # In real implementation, handle errors according to strategy
                continue
        
        return processed_files
    
    return process_ast_files


async def example_basic_usage():
    """Example of basic batch processing usage."""
    logger.info("=== Basic Batch Processing Example ===")
    
    # Create a basic configuration
    config = BatchConfig(
        batch_size=10,
        max_workers=2,
        processing_timeout=60.0,
        batch_strategy=BatchStrategy.COUNT_BASED,
        error_handling_mode=ErrorHandlingMode.CONTINUE
    )
    
    # Create processor function
    processor = create_sample_processor()
    
    # Create engine
    engine = BatchProcessingEngine(
        config=config,
        processor_func=processor
    )
    
    # Process a directory (using sample data)
    source_path = Path("ast_output/Daily/src")
    
    if source_path.exists():
        logger.info(f"Processing directory: {source_path}")
        
        result = await engine.process_directory(
            source_path=source_path,
            resume_from_checkpoint=False
        )
        
        logger.info(f"Processing completed:")
        logger.info(f"  Files discovered: {result.files_discovered}")
        logger.info(f"  Files processed: {result.files_processed}")
        logger.info(f"  Files successful: {result.files_successful}")
        logger.info(f"  Files failed: {result.files_failed}")
        logger.info(f"  Processing time: {result.total_processing_time:.2f}s")
        logger.info(f"  Throughput: {result.throughput_files_per_second:.1f} files/sec")
        logger.info(f"  Batches created: {len(result.batch_results)}")
    else:
        logger.warning(f"Directory not found: {source_path}")


async def example_high_throughput():
    """Example of high-throughput batch processing."""
    logger.info("=== High-Throughput Processing Example ===")
    
    # Use high-throughput configuration
    config = BatchConfig.high_throughput()
    
    # Customize for even higher throughput
    config.max_workers = 8
    config.batch_size = 100
    config.enable_compression = True
    
    processor = create_sample_processor()
    
    engine = BatchProcessingEngine(
        config=config,
        processor_func=processor
    )
    
    source_path = Path("ast_output/Daily/src")
    
    if source_path.exists():
        logger.info(f"High-throughput processing: {source_path}")
        
        start_time = datetime.now()
        result = await engine.process_directory(source_path)
        end_time = datetime.now()
        
        total_time = (end_time - start_time).total_seconds()
        
        logger.info(f"High-throughput results:")
        logger.info(f"  Total time: {total_time:.2f}s")
        logger.info(f"  Files processed: {result.files_processed}")
        logger.info(f"  Throughput: {result.throughput_files_per_second:.1f} files/sec")
        logger.info(f"  Workers used: {config.max_workers}")
        logger.info(f"  Batch size: {config.batch_size}")


async def example_memory_optimized():
    """Example of memory-optimized batch processing."""
    logger.info("=== Memory-Optimized Processing Example ===")
    
    # Use memory-optimized configuration
    config = BatchConfig.memory_optimized()
    
    processor = create_sample_processor()
    
    engine = BatchProcessingEngine(
        config=config,
        processor_func=processor
    )
    
    source_path = Path("ast_output/Daily/src")
    
    if source_path.exists():
        logger.info(f"Memory-optimized processing: {source_path}")
        
        result = await engine.process_directory(source_path)
        
        logger.info(f"Memory-optimized results:")
        logger.info(f"  Files processed: {result.files_processed}")
        logger.info(f"  Batch size: {config.batch_size}")
        logger.info(f"  Memory limit: {config.memory_limit_mb}MB")
        logger.info(f"  Strategy: {config.batch_strategy.value}")


async def example_with_checkpoints():
    """Example of batch processing with checkpoint support."""
    logger.info("=== Checkpoint-Enabled Processing Example ===")
    
    # Create checkpoint manager
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    checkpoint_manager = CheckpointManager(checkpoint_dir)
    
    # Use reliable configuration with checkpointing
    config = BatchConfig.reliable()
    
    processor = create_sample_processor()
    
    engine = BatchProcessingEngine(
        config=config,
        processor_func=processor,
        checkpoint_manager=checkpoint_manager
    )
    
    source_path = Path("ast_output/Daily/src")
    
    if source_path.exists():
        logger.info(f"Processing with checkpoints: {source_path}")
        
        # First run
        result = await engine.process_directory(
            source_path=source_path,
            resume_from_checkpoint=True
        )
        
        logger.info(f"Checkpoint-enabled results:")
        logger.info(f"  Files processed: {result.files_processed}")
        logger.info(f"  Checkpointing enabled: {config.enable_checkpointing}")
        logger.info(f"  Checkpoint interval: {config.checkpoint_interval}")
        
        # List available checkpoints
        checkpoints = await checkpoint_manager.list_checkpoints()
        logger.info(f"  Available checkpoints: {len(checkpoints)}")


async def example_adaptive_batching():
    """Example of adaptive batch sizing based on performance."""
    logger.info("=== Adaptive Batching Example ===")
    
    config = BatchConfig(
        batch_size=50,  # Starting size
        max_workers=4,
        batch_strategy=BatchStrategy.ADAPTIVE,
        processing_timeout=120.0
    )
    
    processor = create_sample_processor()
    
    engine = BatchProcessingEngine(
        config=config,
        processor_func=processor
    )
    
    source_path = Path("ast_output/Daily/src")
    
    if source_path.exists():
        logger.info(f"Adaptive batching processing: {source_path}")
        
        result = await engine.process_directory(source_path)
        
        logger.info(f"Adaptive batching results:")
        logger.info(f"  Files processed: {result.files_processed}")
        logger.info(f"  Batches created: {len(result.batch_results)}")
        logger.info(f"  Strategy: {config.batch_strategy.value}")
        
        # Show batch size variations
        batch_sizes = [len(br.metrics.get('batch_files', [])) for br in result.batch_results if 'batch_files' in br.metrics]
        if batch_sizes:
            logger.info(f"  Batch size range: {min(batch_sizes)} - {max(batch_sizes)}")


async def example_error_handling_strategies():
    """Example of different error handling strategies."""
    logger.info("=== Error Handling Strategies Example ===")
    
    def failing_processor(files: List[ASTFile]) -> List[ASTFile]:
        """Processor that fails on some files for demonstration."""
        processed = []
        for i, file in enumerate(files):
            if i % 5 == 0:  # Fail every 5th file
                raise Exception(f"Simulated failure for {file.path}")
            processed.append(file)
        return processed
    
    # Test different error handling modes
    error_modes = [
        ErrorHandlingMode.CONTINUE,
        ErrorHandlingMode.RETRY,
        ErrorHandlingMode.QUARANTINE
    ]
    
    source_path = Path("ast_output/Daily/src")
    
    if not source_path.exists():
        logger.warning(f"Directory not found: {source_path}")
        return
    
    for error_mode in error_modes:
        logger.info(f"Testing error mode: {error_mode.value}")
        
        config = BatchConfig(
            batch_size=10,
            max_workers=2,
            error_handling_mode=error_mode,
            retry_attempts=3 if error_mode == ErrorHandlingMode.RETRY else 1
        )
        
        engine = BatchProcessingEngine(
            config=config,
            processor_func=failing_processor
        )
        
        try:
            result = await engine.process_directory(source_path)
            
            logger.info(f"  Error mode {error_mode.value} results:")
            logger.info(f"    Files processed: {result.files_processed}")
            logger.info(f"    Files successful: {result.files_successful}")
            logger.info(f"    Files failed: {result.files_failed}")
            
        except Exception as e:
            logger.error(f"  Error mode {error_mode.value} failed: {e}")


async def example_performance_monitoring():
    """Example of monitoring batch processing performance."""
    logger.info("=== Performance Monitoring Example ===")
    
    config = BatchConfig.high_throughput()
    processor = create_sample_processor()
    
    engine = BatchProcessingEngine(
        config=config,
        processor_func=processor
    )
    
    source_path = Path("ast_output/Daily/src")
    
    if source_path.exists():
        logger.info(f"Starting performance monitoring for: {source_path}")
        
        # Start processing in background
        processing_task = asyncio.create_task(
            engine.process_directory(source_path)
        )
        
        # Monitor progress
        while not processing_task.done():
            await asyncio.sleep(2)  # Check every 2 seconds
            
            metrics = await engine.get_metrics()
            
            logger.info(f"Progress update:")
            logger.info(f"  Files processed: {metrics['total_files_processed']}")
            logger.info(f"  Throughput: {metrics['throughput_files_per_second']:.1f} files/sec")
            logger.info(f"  Active workers: {metrics['worker_count']}")
            logger.info(f"  Batches completed: {metrics['batch_count']}")
        
        # Get final result
        result = await processing_task
        
        logger.info(f"Final performance metrics:")
        logger.info(f"  Total processing time: {result.total_processing_time:.2f}s")
        logger.info(f"  Average throughput: {result.throughput_files_per_second:.1f} files/sec")
        logger.info(f"  Total batches: {len(result.batch_results)}")


async def example_custom_configuration():
    """Example of creating custom batch processing configurations."""
    logger.info("=== Custom Configuration Example ===")
    
    # Create a custom configuration for specific requirements
    custom_config = BatchConfig(
        # Processing parameters
        batch_size=75,
        max_workers=6,
        processing_timeout=180.0,
        
        # Strategy and error handling
        batch_strategy=BatchStrategy.SIZE_BASED,
        error_handling_mode=ErrorHandlingMode.RETRY,
        retry_attempts=2,
        
        # Performance tuning
        memory_limit_mb=1024,
        enable_compression=True,
        
        # Checkpointing
        enable_checkpointing=True,
        checkpoint_interval=25,
        
        # Queue management
        max_queue_size=200,
        
        # File size limits
        batch_size_mb=10  # Size-based batching parameter
    )
    
    processor = create_sample_processor()
    
    engine = BatchProcessingEngine(
        config=custom_config,
        processor_func=processor
    )
    
    source_path = Path("ast_output/Daily/src")
    
    if source_path.exists():
        logger.info(f"Custom configuration processing: {source_path}")
        
        result = await engine.process_directory(source_path)
        
        logger.info(f"Custom configuration results:")
        logger.info(f"  Configuration: Custom optimized")
        logger.info(f"  Files processed: {result.files_processed}")
        logger.info(f"  Processing time: {result.total_processing_time:.2f}s")
        logger.info(f"  Throughput: {result.throughput_files_per_second:.1f} files/sec")
        logger.info(f"  Batch strategy: {custom_config.batch_strategy.value}")
        logger.info(f"  Error handling: {custom_config.error_handling_mode.value}")


async def main():
    """Run all batch processing examples."""
    logger.info("Starting Batch Processing Engine Examples")
    logger.info("=" * 50)
    
    examples = [
        ("Basic Usage", example_basic_usage),
        ("High Throughput", example_high_throughput),
        ("Memory Optimized", example_memory_optimized),
        ("With Checkpoints", example_with_checkpoints),
        ("Adaptive Batching", example_adaptive_batching),
        ("Error Handling", example_error_handling_strategies),
        ("Performance Monitoring", example_performance_monitoring),
        ("Custom Configuration", example_custom_configuration)
    ]
    
    for name, example_func in examples:
        try:
            logger.info(f"\n{'=' * 20} {name} {'=' * 20}")
            await example_func()
            logger.info(f"✅ {name} completed successfully")
            
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
        
        # Small delay between examples
        await asyncio.sleep(1)
    
    logger.info("\n" + "=" * 50)
    logger.info("All Batch Processing Examples Completed")


if __name__ == "__main__":
    asyncio.run(main())
