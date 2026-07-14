import argparse
import logging
import os
import sys
import time
import json

from pathlib import Path
import yaml

parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from data_handler import DataHandler
from roc import roc
from score_distribution import scores


def setup_logger(out_dir: Path, unique_id: str) -> logging.Logger:
    """Creates and configures a global logger that all functions can access."""
    os.makedirs(out_dir, exist_ok=True)  # Ensure log directory exists

    # Unique log filename per process
    process_id = os.getpid()
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    log_filename = out_dir / f'eval_{time_stamp}.log'

    # Create a named logger (global use)
    logger = logging.getLogger(f"{unique_id}")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # file_handler = logging.FileHandler(log_filename)
        file_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)

    return logger


def main(args: argparse.Namespace):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # setup logger
    logger = setup_logger(out_dir, 'eval')

    # load the configuration file, check if file exists
    config_file = args.config
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"[*] --> Configuration file not found at {config_file}")
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
        logger.info(f"[*] --> Configuration file loaded from {config_file}")

    # load the data
    input_data = config['input_data']
    data_handlers = []
    for data in input_data:
        data_handler = DataHandler(
            data, logger=setup_logger(out_dir, 'DataHandler'),
            hhard_binning=config['score_distribution']['trafo'].get('use_hhard_binning', False)
        )

        if args.eval_fold == 'eval_all':
            for fold in data_handler.fold_dirs:
                data_handler.load_data(fold)
            data_handler.combine()
        else:
            data_handler.load_data(args.eval_fold)

        data_handler.convert_to_hist(
            x_min=config['score_distribution']['x_min'],
            x_max=config['score_distribution']['x_max'],
        )

        # print statistics
        data_handler.print_statistics()

        data_handlers.append(data_handler)

    # merge all data_handlers
    data_handler = DataHandler.concat(data_handlers)

    # save histograms to disk
    hist_folder = out_dir / 'histograms'
    hist_folder.mkdir(parents=True, exist_ok=True)
    if args.create_hists:
        data_handler.save_to_root(hist_folder, logger=logger)

    # plot ROC
    logger.info(f"[*] --> Running ROCs")

    # New update
    if 'roc_range' not in config['roc'] and 'reference_dataset' not in config['roc'] and 'reference_classifier' not in \
            config['roc'] and 'reference_fold' not in config['roc']:
        print(
            '[*] --> WARNING: please update your config file to include the new roc_range, reference_dataset, reference_classifier and reference_fold')
        print('[*] --> Check the latest config file for the new parameters')
        print(f""" Or you can use the following example:
        
--> Your eval_config.yaml <--
roc:
  drop_negative_weights: True
  figure_size: [ 10, 8 ]
  # All configs bellow are ONLY for background rejection vs signal efficiency plot
  roc_range: [ 0.1, 1.0 ]
  reference_dataset: 'test' # dataset used for reference in ratio plot
  reference_classifier: "BDT ggF High" # classifier used for referece in ratio plot
  reference_fold: 'fold_0' # fold used for reference in ratio plot
""")
        exit(1)

    roc_result = roc(
        data_handler,
        roc_range=config['roc']['roc_range'],
        reference_dataset=config['roc']['reference_dataset'],
        reference_classifier=config['roc']['reference_classifier'],
        reference_fold=config['roc']['reference_fold'],
        drop_negative_weights=config['roc']['drop_negative_weights'],
        figsize=config['roc']['figure_size'],
        save_path=out_dir,
        logger=logger,
        include_roc=config.get('include_roc', True),
        include_roc_inverse=config.get('include_roc_inverse', True),
    )

    # save the results
    roc_file = out_dir / 'roc.json'
    with open(roc_file, 'w') as f:
        json.dump(roc_result, f, indent=4)

    logger.info(f"[*] --> Results saved to {roc_file}")

    # without trafo
    if config.get('include_score_distribution', True):
        logger.info(f"[*] --> Running Score Distribution - without trafo")
        sig_result = scores(
            data_handler,
            category=config['score_distribution']['category'],
            trafo=None,
            figsize=config['score_distribution']['figure_size'],
            bin_num=config['score_distribution']['bin_num'],
            x_min=config['score_distribution']['x_min'],
            x_max=config['score_distribution']['x_max'],
            y_log=config['score_distribution']['y_log'],
            reference_dataset=config['score_distribution']['reference_dataset'],
            baseline_classifier=config['score_distribution']['baseline_classifier'],
            ratio_pulls=config['score_distribution']['ratio_pulls'],
            fold_colors=config['score_distribution']['fold_colors'],
            fold_reference=config['score_distribution']['fold_reference'],
            save_path=out_dir,
            logger=logger,
        )
        sig_file = out_dir / 'sig_without_trafo.json'
        with open(sig_file, 'w') as f:
            json.dump(sig_result, f, indent=4)

        logger.info(f"[*] --> Results saved to {sig_file}")

    # with trafo
    if config.get('include_score_distribution_trafo', True):
        logger.info(f"[*] --> Running Score Distribution - with trafo")
        sig_result = scores(
            data_handler,
            category=config['score_distribution']['category'],
            trafo=config['score_distribution']['trafo'],
            figsize=config['score_distribution']['figure_size'],
            bin_num=config['score_distribution']['bin_num'],
            x_min=config['score_distribution']['x_min'],
            x_max=config['score_distribution']['x_max'],
            y_log=config['score_distribution']['y_log'],
            reference_dataset=config['score_distribution']['reference_dataset'],
            baseline_classifier=config['score_distribution']['baseline_classifier'],
            ratio_pulls=config['score_distribution']['ratio_pulls'],
            fold_colors=config['score_distribution']['fold_colors'],
            fold_reference=None,
            save_path=out_dir,
            logger=logger,
            use_test_bin_edges=config['score_distribution'].get('use_test_bin_edges', False),
        )

        sig_file = out_dir / 'sig_with_trafo.json'
        with open(sig_file, 'w') as f:
            json.dump(sig_result, f, indent=4)

        logger.info(f"[*] --> Results saved to {sig_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate the performance of the SALT model for bbtautau Run 2 + Run 3 Analysis'
    )

    parser.add_argument('config', type=str, help='Path to the configuration file')
    parser.add_argument('--out_dir', type=str, default='./', help='Output directory')
    parser.add_argument('--eval_fold', type=str, default='eval_all',
                        help='Fold to evaluate [eval_all to eval all folds]')
    parser.add_argument('--create_hists', action='store_true', help='create output histograms for validation')

    args = parser.parse_args()

    main(args)
