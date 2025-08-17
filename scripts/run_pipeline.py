#!/usr/bin/env python3
"""
CLI entry point for running snake-pipe ETL pipelines
"""

import argparse
import sys
from pathlib import Path

from snake_pipe.pipeline import Pipeline, run_pipeline
from snake_pipe.utils.logger import get_logger

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


logger = get_logger(__name__)


def main() -> None:
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Snake Pipe - Python ETL Pipeline Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run default pipeline
  python run_pipeline.py

  # Run with custom configuration
  python run_pipeline.py --config config.yaml

  # Run specific pipeline
  python run_pipeline.py --pipeline my_pipeline
        """,
    )

    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")

    parser.add_argument("--pipeline", "-p", type=str, default="default", help="Pipeline name to run (default: default)")

    parser.add_argument("--input", "-i", type=str, help="Input file path")

    parser.add_argument("--output", "-o", type=str, help="Output file path")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    try:
        if args.verbose:
            logger.setLevel("DEBUG")

        logger.info("Starting snake-pipe CLI")
        logger.info(f"Pipeline: {args.pipeline}")

        if args.input and args.output:
            # Run simple file transformation pipeline
            logger.info(f"Input: {args.input}")
            logger.info(f"Output: {args.output}")

            pipeline = Pipeline(f"cli_{args.pipeline}")

            # Determine input type based on file extension
            input_path = Path(args.input)
            output_path = Path(args.output)

            if input_path.suffix.lower() == ".csv":
                pipeline.extract_from_csv(str(input_path))
            else:
                raise ValueError(f"Unsupported input file type: {input_path.suffix}")

            # Add basic cleaning
            pipeline.clean_data()

            # Determine output type based on file extension
            if output_path.suffix.lower() == ".csv":
                pipeline.load_to_csv(output_path.name, str(output_path.parent))
            else:
                raise ValueError(f"Unsupported output file type: {output_path.suffix}")

            # Run the pipeline
            result = pipeline.run()
            logger.info(f"Pipeline completed successfully. Processed {len(result)} rows.")

        else:
            # Run default pipeline
            run_pipeline()

        logger.info("CLI execution completed successfully")

    except Exception as e:
        logger.error(f"CLI execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
