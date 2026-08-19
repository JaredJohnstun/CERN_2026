import copy

import h5py
import numpy as np
import pandas as pd
import re
import logging
from pathlib import Path
from array import array
import yaml
import uproot as up

def get_hhard_binning_array():
    bins = array('d')
    for i in range(1990):
        bins.append(round(i / 1000. - 1.0, 3))
    for i in range(101):
        bins.append(round(0.99 + i / 10000.0, 4))
    return bins


def match_file(file_path: Path, pattern: str):
    epoch_num, val_loss, dataset_type, original_filename = None, None, None, None

    match = re.match(pattern, file_path.name)
    if match:
        epoch_num = int(match.group(1))
        val_loss = float(match.group(2))
        dataset_type = match.group(3)
        try:
            original_filename = match.group(4)
        except IndexError:
            original_filename = None

    return {
        'epoch_num': epoch_num,
        'val_loss': val_loss,
        'dataset_type': dataset_type,
        'original_filename': original_filename
    }


class DataHandler:
    def __init__(self, config: dict, logger: logging.Logger, hhard_binning: bool = False):
        self.base_dir = Path(config['base_dir'])
        self.fold_dirs = {
            k: self.base_dir / f for k, f in config['fold_dirs'].items()
        }
        self.file_pattern = config['file_pattern']
        self.file_block = config['file_block']
        self.weight_col = config['weight_col']
        self.label_col = config['label_col']
        self.classifier_config = config['classifiers']
        self.train_config = config['train_config']
        self.meta_data = config['meta_data']
        self.hhard_bin = hhard_binning
        self.default_bin_edges = None

        self.dataset_types = ['train', 'val', 'test']

        self.fold_data = {
            k: {
                ds_type: {
                    classifier: pd.DataFrame()
                    for classifier in self.classifier_config
                }
                for ds_type in self.dataset_types
            } for k in self.fold_dirs
        }

        self.combined_data = {
            ds_type: {
                classifier: pd.DataFrame()
                for classifier in self.classifier_config
            }
            for ds_type in self.dataset_types
        }

        self.base_bin_num = config['base_bin_num']
        self.hists = {
            score: {
                k: {
                    ds_type: {'hist': pd.DataFrame(), 'bin_edges': []} for ds_type in self.dataset_types
                } for k in [*self.fold_dirs.keys(), 'combined']
            } for score in self.classifier_config
        }

        self.logger = logger

    @classmethod
    def concat(cls, handlers):
        """Concatenates multiple DataHandler instances."""
        if len(handlers) < 1:
            raise ValueError("At least one DataHandler instance is required")

        # Ensure all inputs are DataHandler instances
        for h in handlers:
            if not isinstance(h, cls):
                raise TypeError(f"Expected instances of {cls.__name__}, got {type(h)}")

        # Create a new instance based on the first one (deepcopy to avoid modification)
        merged_data_handler = copy.deepcopy(handlers[0])

        merged_data_handler.base_dir = None
        merged_data_handler.file_pattern = None
        merged_data_handler.file_block = None
        merged_data_handler.weight_col = 'weight'
        merged_data_handler.label_col = 'label'
        merged_data_handler.train_config = None
        merged_data_handler.meta_data = None
        # merged_data_handler.hhard_bin = True

        merged_data_handler.fold_dirs = {k: None for k in merged_data_handler.fold_dirs.keys()}
        # merged_data_handler.fold_data = None
        # merged_data_handler.combined_data = None

        if len(handlers) == 1:
            return merged_data_handler

        for h in handlers[1:]:
            merged_data_handler.classifier_config.update(h.classifier_config)
            merged_data_handler.fold_dirs.update(h.fold_dirs)

            # Merge fold data
            for fold, data in h.fold_data.items():
                for data_type in merged_data_handler.dataset_types:
                    merged_data_handler.fold_data[fold][data_type].update(data[data_type])

            # merge combined data
            for data_type in merged_data_handler.dataset_types:
                merged_data_handler.combined_data[data_type].update(h.combined_data[data_type])

            # Merge Histograms
            merged_data_handler.hists.update(h.hists)

        return merged_data_handler

    def _load_scores(
            self,
            file_path: Path,
            dataset_data: dict[str, pd.DataFrame],
            model_name: str,
    ) -> None:

        def decode_name(hi: int, lo: int) -> str:
            # re-assemble the 16 bytes and strip padding
            b = hi.to_bytes(8, byteorder='big') + lo.to_bytes(8, byteorder='big')
            return b.rstrip(b'\0').decode('ascii')

        with h5py.File(file_path, 'r') as f:
            data = pd.DataFrame(f[self.file_block][:])
            # find all columsn that are string (i.e. with hi and lo columns)
            hi_cols = [c for c in data.columns if c.endswith('_hi')]
            strs    = [c[:-3] for c in hi_cols if c[:-3] + '_lo' in data.columns]

            for istr in strs:
                hi_c = istr + '_hi'
                lo_c = istr + '_lo'
                # decode and store string as it's own column
                data[istr] = [decode_name(h, l) for h, l in zip(data[hi_c], data[lo_c])]
                data.drop(columns=[hi_c, lo_c], inplace=True) # drop the hi/lo columns
            self.has_sample_name = 'sample_name' in data.columns
            if self.has_sample_name:
                unique = data.drop_duplicates(subset='sample_name', keep='first')
                self.sample_to_label = dict(
                    zip(unique['sample_name'], unique[self.label_col])
                )
            else:
                self.logger.warn(
                    "No sample_name present; "
                    "significance calc will differ from WSMaker"
                )

            for score_name, settings in self.classifier_config.items():
                columns = settings["columns"]
                expression = settings["expression"]
                selection = settings.get("selection", None)

                # Prepare local variables for eval()
                local_vars = {}
                for var_name, column_template in columns.items():
                    column_name = column_template.format(model_name=model_name)  # Replace {name} dynamically
                    if column_name in data.columns:
                        local_vars[var_name] = data[column_name]
                    else:
                        raise KeyError(f"Column '{column_name}' not found in the input file.")

                # Add numpy and custom functions to local_vars
                local_vars.update({"np": np})

                # Evaluate the expression and store in the dataframe
                data[score_name] = eval(expression, {}, local_vars)

                if selection is not None:
                    flag = eval(selection, {}, local_vars)
                    selected_data = data[flag].copy()
                else:
                    selected_data = data
                
                if self.has_sample_name:
                    dataset_data[score_name] = selected_data[[self.weight_col, self.label_col, score_name, 'mcChannelNumber', 'sample_name']].copy()
                    dataset_data[score_name].rename(columns={self.weight_col: 'weight', self.label_col: 'label'}, inplace=True)
                else:
                    dataset_data[score_name] = selected_data[[self.weight_col, self.label_col, score_name, 'mcChannelNumber']].copy()
                    dataset_data[score_name].rename(columns={self.weight_col: 'weight', self.label_col: 'label'}, inplace=True)

            

    def load_data(self, fold: str):
        if fold not in self.fold_dirs:
            raise ValueError(f"[*] --> Invalid fold: {fold}\n Available folds: {self.fold_dirs.keys()}")

        fold_dir = self.fold_dirs[fold]

        # load train configs (check if file exists, then load and print)
        train_config_file = fold_dir / self.train_config
        if not train_config_file.exists():
            raise FileNotFoundError(f"[*] --> Train config file not found at {train_config_file}")
        with open(train_config_file, 'r') as f:
            train_config = yaml.safe_load(f)
            model_name = train_config['name']
            self.logger.info(f"[*] --> Model name {model_name} loaded from {train_config_file}")

        for files in (fold_dir / 'ckpts').iterdir():
            if files.is_file():
                file_info = match_file(files, self.file_pattern)

                for data_type in ['train', 'val', 'test']:
                    if file_info['dataset_type'] == data_type:
                        self._load_scores(files, self.fold_data[fold][data_type], model_name=model_name)
                        break

    def combine(self):
        self.logger.info("[*] --> Combining data from all folds")

        for fold, data in self.fold_data.items():
            for data_type in ['train', 'val', 'test']:
                for score in self.classifier_config:
                    if not data[data_type][score].empty:
                        self.combined_data[data_type][score] = pd.concat(
                            [self.combined_data[data_type][score], data[data_type][score]]
                        )

    def convert_to_hist(self, x_min: float = 0.0, x_max: float = 1.0):
        self.logger.info("[*] --> Converting data to histograms...")

        if not isinstance(x_min, (int, float)) or not isinstance(x_max, (int, float)):
            raise ValueError("x_min and x_max must be integers or floats")
        if x_min >= x_max:
            raise ValueError("x_min must be less than x_max")

        data = self.combined_data
        # empty = sum([data_df.empty for data_df in data.values()]) == len(data)
        empty = all([data_df.empty for scores in data.values() for data_df in scores.values()])

        loop_data = self.fold_data
        if not empty:
            loop_data = {**loop_data, 'combined': data}

        bin_edges = np.linspace(x_min, x_max, self.base_bin_num + 1)
        self.default_bin_edges = bin_edges
        if self.hhard_bin:
            bin_edges = np.array(get_hhard_binning_array())
            self.logger.info(f"[*] --> Using HHARD binning of {bin_edges}")
        else:
            self.logger.info(f"[*] --> Bin edges: {x_min} to {x_max}, {self.base_bin_num} bins")

        for score in self.hists.keys():
            for fold, data in loop_data.items():
                for data_type, score_df in data.items():
                    data_df = score_df[score]
                    if data_df.empty:
                        continue

                    labels = data_df['label'].unique()

                    dfs = []
                    for label in labels:
                        if self.has_sample_name:
                            data_label = data_df[data_df['label'] == label].drop(columns="sample_name").astype(np.float64)
                        else:
                            data_label = data_df[data_df['label'] == label].astype(np.float64)
                        weights = data_label['weight'].astype(np.float64)
                        hist, _ = np.histogram(data_label[score], bins=bin_edges, weights=weights)
                        unc, _ = np.histogram(data_label[score], bins=bin_edges, weights=weights ** 2)
                        unc = np.sqrt(unc)

                        dfs.append(pd.DataFrame({
                            f'{label}': hist,
                            f'{label}_unc': unc
                        }))

                    for channel in data_df['mcChannelNumber'].unique():
                        if self.has_sample_name:
                            chan_df = data_df[data_df['mcChannelNumber'] == channel].drop(columns="sample_name").astype(np.float64)
                        else:
                            chan_df = data_df[data_df['mcChannelNumber'] == channel].astype(np.float64)
                        w_chan = chan_df['weight']
                        h_chan, _ = np.histogram(chan_df[score], bins=bin_edges, weights=w_chan)
                        u_chan, _ = np.histogram(chan_df[score], bins=bin_edges, weights=w_chan**2)
                        u_chan = np.sqrt(u_chan)
                        dfs.append(pd.DataFrame({
                            f'chan{int(channel)}':     h_chan,
                            f'chan{int(channel)}_unc': u_chan
                        }))

                    if self.has_sample_name:
                        self.unique_sample_list = data_df['sample_name'].unique()
                        for sample in self.unique_sample_list:
                            samp_df = data_df[data_df['sample_name'] == sample] #.astype(h5py.string_dtype(encoding="utf-8"))
                            w_samp = samp_df['weight']
                            h_samp, _ = np.histogram(samp_df[score], bins=bin_edges, weights=w_samp)
                            u_samp, _ = np.histogram(samp_df[score], bins=bin_edges, weights=w_samp**2)
                            u_samp = np.sqrt(u_samp)
                            dfs.append(pd.DataFrame({
                                f'{sample}':     h_samp,
                                f'{sample}_unc': u_samp
                            }))

                    self.hists[score][fold][data_type]['hist'] = pd.concat(dfs, axis=1)
                    self.hists[score][fold][data_type]['bin_edges'] = bin_edges
                    self.hists[score][fold][data_type]['yields'] = self.hists[score][fold][data_type]['hist'].sum(
                        axis=0)

    def print_statistics(self):
        out_str = "\n📊 Model Training Statistics Summary 📊\n"

        for fold, data in {**self.fold_data, 'Combined(*)': self.combined_data}.items():
            out_str += f"\n🔥 {fold.upper()} 🔥 \n"
            stats_list = []

            # Collect all unique labels across all datasets
            all_labels = set()
            for df in data.values():
                for score in df.values():
                    if not score.empty:
                        all_labels.update(score['label'].unique())

            # Process each data split
            for data_type in ["train", "val", "test"]:
                # df = data.get(data_type, pd.DataFrame()) # Get DataFrame, default to empty
                df = list(data.get(data_type).values())[0]

                for label in all_labels:
                    if df.empty or label not in df['label'].values:
                        total_entries, total_yields = 0, 0  # Default to 0
                    else:
                        df_filtered = df[df['label'] == label]
                        total_entries = len(df_filtered)
                        total_yields = df_filtered['weight'].sum()

                    stats_list.append([data_type, label, total_entries, total_yields])

            # Convert to Pandas DataFrame for tabular display
            stats_df = pd.DataFrame(stats_list, columns=["Data Split", "Label", "Entries", "Yields"])
            stats_df = stats_df.pivot(index="Label", columns="Data Split", values=["Entries", "Yields"])

            # Sort the columns by these orders
            data_splits = ['train', 'val', 'test']
            level0_order = ['Entries', 'Yields']
            stats_df = stats_df.loc[:, sorted(
                stats_df.columns,
                key=lambda x: (level0_order.index(x[0]), data_splits.index(x[1]))
            )]

            # Define column widths for formatting
            col_width = 13  # Adjust column width as needed
            label_width = 8  # Width for label column

            # Prepare multi-line header
            header_line1 = f"{'Label':<{label_width}}" + "".join(
                [f"{'Entries':^{col_width * 3}}{'Yields':^{col_width * 3}}"])
            header_line2 = f"{'':<{label_width}}" + "".join([f"{dtype:^{col_width}}" for dtype in data_splits] * 2)

            # Print the header
            out_str += f"{header_line1}\n{header_line2}\n"
            out_str += ("-" * len(header_line1)) + "\n"

            # Print formatted table rows
            for label, row in stats_df.iterrows():
                row_str = f"{label:<{label_width}}"  # Label column
                for col in stats_df.columns:
                    value = row[col]
                    if value == 0 or pd.isna(value):
                        row_str += f"{'-':^{col_width}}"  # Empty entries formatted as "-"
                    else:
                        if "Entries" in col:
                            row_str += f"{value:^{col_width},.0f}"
                        else:
                            row_str += f"{value:^{col_width}.2f}"
                out_str += row_str + "\n"

        self.logger.info(out_str)

    def get_fold_names(self):
        keys = self.fold_dirs.keys()

        if not all([score.empty for dt in self.combined_data.values() for score in dt.values()]):
            keys = [*keys, 'combined']

        return keys

    def get_classifier_names(self):
        return self.classifier_config.keys()

    def save_to_root(self, out_dir: Path, logger: logging.Logger = None):
        with up.recreate(out_dir / 'data.root') as f:
            for fold in self.get_fold_names():
                for data_type in self.dataset_types:
                    for score in self.get_classifier_names():
                        hist_data = self.hists[score][fold][data_type]['hist']
                        bin_edges = self.hists[score][fold][data_type]['bin_edges']

                        if hist_data.empty:
                            continue

                        class_list = [col for col in hist_data.columns.tolist() if not col.endswith('_unc')]
                        for class_name in class_list:
                            hist_values = hist_data[class_name].values

                            # Create a histogram in ROOT format
                            hist_name = f"{fold}_{data_type}_{score}_{class_name}".replace(' ', '')
                            f[hist_name] = (hist_values, bin_edges)

                            if logger:
                                logger.info(f"[*] --> Saved histogram {hist_name} to ROOT file {out_dir / 'data.root'}")
