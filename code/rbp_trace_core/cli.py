"""
cli.py

Command-line interface for running RBP_TRACE (Joint Protein-Ligand Embedding).

"""
import logging

from .parser import build_parser
from .run_rbp_trace_core import main as run_rbp_trace_main

def main_cli() -> None:
    """
    Main entry point for the RBP_TRACE CLI.

    Parse command-line arguments, validate inputs, and run the RBP_TRACE program.

    """

    # Configure logging behavior
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Parse arguments
    parser = build_parser()
    args = parser.parse_args()

    # Validation
    if args.mode != 'predict_na' and args.fasta_path is None:
        parser.error('--fasta is required when mode is "train" or '
                     '"predict_protein".')
    if args.mode != 'predict_protein' and args.y_path is None:
        parser.error('--zscore is required when mode is "train" or '
                     '"predict_na".')

    # Call the core API
    run_rbp_trace_main(
        mode=args.mode,
        param_path=args.param_path,
        fasta_path=args.fasta_path,
        hmm_path=args.hmm_path,
        y_path=args.y_path,
        output_path=args.output_path,
        return_results=False
    )


if __name__ == '__main__':
    main_cli()
