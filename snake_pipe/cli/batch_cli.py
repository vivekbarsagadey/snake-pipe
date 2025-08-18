"""Command-line interface for batch processing operations.

This module provides a CLI for running batch processing operations
with various configurations and options.
"""

import argparse
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from snake_pipe.extract.batch_config import (
    BatchConfig, BatchStrategy, ErrorHandlingMode, ProcessingMode
)
from snake_pipe.extract.batch_processor import BatchProcessingEngine
from snake_pipe.extract.models import ASTFile
from snake_pipe.utils.checkpoint_manager import CheckpointManager


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('batch_processing.log')
        ]
    )


def create_sample_processor():
    """Create a sample processor function for CLI usage."""
    def process_files(files: List[ASTFile]) -> List[ASTFile]:
        """Basic processor that simulates file processing."""
        processed = []
        for file in files:
            # Simulate processing
            import time
            time.sleep(0.001)  # 1ms per file
            
            # Mark as processed
            file.metadata.processed_time = datetime.now()
            processed.append(file)
        
        return processed
    
    return process_files


def create_config_from_args(args) -> BatchConfig:
    """Create BatchConfig from command line arguments."""
    # Map string values to enums
    strategy_map = {
        'count': BatchStrategy.COUNT_BASED,
        'size': BatchStrategy.SIZE_BASED,
        'time': BatchStrategy.TIME_BASED,
        'adaptive': BatchStrategy.ADAPTIVE
    }
    
    error_mode_map = {
        'fail-fast': ErrorHandlingMode.FAIL_FAST,
        'continue': ErrorHandlingMode.CONTINUE,
        'retry': ErrorHandlingMode.RETRY,
        'quarantine': ErrorHandlingMode.QUARANTINE
    }
    
    return BatchConfig(
        batch_size=args.batch_size,
        max_workers=args.workers,
        processing_timeout=args.timeout,
        batch_strategy=strategy_map.get(args.strategy, BatchStrategy.COUNT_BASED),
        error_handling_mode=error_mode_map.get(args.error_mode, ErrorHandlingMode.CONTINUE),
        retry_attempts=args.retries,
        memory_limit_mb=args.memory_limit,
        enable_compression=args.compression,
        enable_checkpointing=args.checkpoints,
        checkpoint_interval=args.checkpoint_interval,
        batch_size_mb=args.batch_size_mb
    )


async def run_batch_processing(args) -> None:
    """Run batch processing with the given arguments."""
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting batch processing CLI")
    logger.info(f"Source: {args.source}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Strategy: {args.strategy}")
    
    # Create configuration
    config = create_config_from_args(args)
    
    # Create checkpoint manager if enabled
    checkpoint_manager = None
    if args.checkpoints:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        checkpoint_manager = CheckpointManager(checkpoint_dir)
        logger.info(f"Checkpointing enabled: {checkpoint_dir}")
    
    # Create processor
    processor = create_sample_processor()
    
    # Create engine
    engine = BatchProcessingEngine(
        config=config,
        processor_func=processor,
        checkpoint_manager=checkpoint_manager
    )
    
    # Run processing
    try:
        source_path = Path(args.source)
        
        if args.files:
            # Process specific files
            file_paths = [Path(f) for f in args.files]
            result = await engine.process_files(
                file_paths=file_paths,
                resume_from_checkpoint=args.resume
            )
        else:
            # Process directory
            result = await engine.process_directory(
                source_path=source_path,
                resume_from_checkpoint=args.resume
            )
        
        # Output results
        logger.info("Processing completed successfully!")
        logger.info(f"Files discovered: {result.files_discovered}")
        logger.info(f"Files processed: {result.files_processed}")
        logger.info(f"Files successful: {result.files_successful}")
        logger.info(f"Files failed: {result.files_failed}")
        logger.info(f"Processing time: {result.total_processing_time:.2f}s")
        logger.info(f"Throughput: {result.throughput_files_per_second:.1f} files/sec")
        logger.info(f"Batches created: {len(result.batch_results)}")
        
        # Save results to file if requested
        if args.output:
            output_data = {
                "execution_id": result.execution_id,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "files_discovered": result.files_discovered,
                "files_processed": result.files_processed,
                "files_successful": result.files_successful,
                "files_failed": result.files_failed,
                "total_processing_time": result.total_processing_time,
                "throughput_files_per_second": result.throughput_files_per_second,
                "batch_count": len(result.batch_results),
                "config": config.__dict__
            }
            
            output_path = Path(args.output)
            with open(output_path, 'w') as f:
                json.dump(output_data, f, indent=2, default=str)
            
            logger.info(f"Results saved to: {output_path}")
        
        # Print summary
        if not args.quiet:
            print(f"\n{'='*50}")
            print("BATCH PROCESSING SUMMARY")
            print(f"{'='*50}")
            print(f"Files processed: {result.files_successful}/{result.files_processed}")
            print(f"Success rate: {(result.files_successful/result.files_processed)*100:.1f}%")
            print(f"Processing time: {result.total_processing_time:.2f}s")
            print(f"Throughput: {result.throughput_files_per_second:.1f} files/sec")
            print(f"Batches: {len(result.batch_results)}")
            print(f"Workers: {args.workers}")
            print(f"Strategy: {args.strategy}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        sys.exit(1)


def create_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Batch Processing Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m batch_cli --source ast_output/Daily/src

  # High throughput with 8 workers
  python -m batch_cli --source data/ --workers 8 --batch-size 100

  # Size-based batching with 5MB batches
  python -m batch_cli --source data/ --strategy size --batch-size-mb 5

  # With checkpointing and resume
  python -m batch_cli --source data/ --checkpoints --resume

  # Process specific files
  python -m batch_cli --files file1.json file2.json file3.json

  # Memory optimized processing
  python -m batch_cli --source data/ --memory-limit 512 --workers 2

  # Adaptive batching with retry on errors
  python -m batch_cli --source data/ --strategy adaptive --error-mode retry
        """
    )
    
    # Input options
    parser.add_argument(
        "--source", "-s",
        type=str,
        help="Source directory to process"
    )
    
    parser.add_argument(
        "--files", "-f",
        nargs="+",
        help="Specific files to process (instead of directory)"
    )
    
    # Processing options
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )
    
    parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=50,
        help="Number of files per batch (default: 50)"
    )
    
    parser.add_argument(
        "--batch-size-mb",
        type=int,
        default=10,
        help="Batch size in MB for size-based strategy (default: 10)"
    )
    
    parser.add_argument(
        "--strategy",
        choices=["count", "size", "time", "adaptive"],
        default="count",
        help="Batching strategy (default: count)"
    )
    
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Processing timeout in seconds (default: 300)"
    )
    
    # Error handling
    parser.add_argument(
        "--error-mode",
        choices=["fail-fast", "continue", "retry", "quarantine"],
        default="continue",
        help="Error handling mode (default: continue)"
    )
    
    parser.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Number of retry attempts (default: 3)"
    )
    
    # Performance options
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=1024,
        help="Memory limit per worker in MB (default: 1024)"
    )
    
    parser.add_argument(
        "--compression",
        action="store_true",
        help="Enable compression for better performance"
    )
    
    # Checkpointing
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Enable checkpointing for resumable processing"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for checkpoint files (default: checkpoints)"
    )
    
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=100,
        help="Checkpoint every N files (default: 100)"
    )
    
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Save results to JSON file"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output summary"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    # Configuration presets
    parser.add_argument(
        "--preset",
        choices=["high-throughput", "memory-optimized", "reliable"],
        help="Use predefined configuration preset"
    )
    
    return parser


def apply_preset(args, preset: str) -> None:
    """Apply predefined configuration preset."""
    if preset == "high-throughput":
        args.workers = 8
        args.batch_size = 200
        args.strategy = "adaptive"
        args.compression = True
        args.error_mode = "continue"
        
    elif preset == "memory-optimized":
        args.workers = 2
        args.batch_size = 25
        args.strategy = "size"
        args.memory_limit = 512
        args.compression = True
        
    elif preset == "reliable":
        args.workers = 4
        args.batch_size = 50
        args.strategy = "count"
        args.error_mode = "retry"
        args.retries = 5
        args.checkpoints = True
        args.checkpoint_interval = 50


async def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Apply preset if specified
    if args.preset:
        apply_preset(args, args.preset)
    
    # Validate arguments
    if not args.source and not args.files:
        parser.error("Must specify either --source or --files")
    
    if args.source and args.files:
        parser.error("Cannot specify both --source and --files")
    
    # Run batch processing
    await run_batch_processing(args)


if __name__ == "__main__":
    asyncio.run(main())
