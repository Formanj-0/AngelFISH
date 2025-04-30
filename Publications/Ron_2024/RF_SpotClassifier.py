import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestClassifier
import os
import glob

class ManualLabeler:
    def __init__(self, crops_path, meta_path, meta_df, output_path=None):
        self.crops = np.load(crops_path)
        self.meta = meta_df.copy()
        self.meta_path = meta_path
        self.replica_name = os.path.basename(meta_path).replace("_all_crop_metadata.csv", "")
        self.output_path = output_path or meta_path

        if "manual_label" not in self.meta.columns:
            self.meta["manual_label"] = np.nan

        self.batch_size = 25
        self.fig, self.axes = None, None
        self.rects = {}
        self.labels = {}
        self.indices = []

    def get_next_batch_indices(self):
        return self.meta[self.meta["manual_label"].isna()].index[:self.batch_size]

    def onclick(self, event):
        for i, ax in enumerate(self.axes.flat):
            if event.inaxes == ax:
                idx = self.indices[i]
                if event.button == 1:
                    self.labels[idx] = 1
                elif event.button == 3:
                    self.labels[idx] = 0
                self.update_border(i, self.labels[idx])
                break

    def update_border(self, plot_idx, label):
        ax = self.axes.flat[plot_idx]
        if self.rects.get(plot_idx):
            self.rects[plot_idx].remove()
        color = "green" if label == 1 else "red"
        self.rects[plot_idx] = ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                                      fill=False, edgecolor=color, linewidth=3))
        idx = self.indices[plot_idx]
        pred_str = self.meta.at[idx, 'rf_prediction'] if 'rf_prediction' in self.meta.columns else ""
        label_str = "[SPOT]" if label == 1 else "[NOT]"
        ax.set_title(f"Idx {idx}\nPRED: {pred_str} → {label_str}")
        self.fig.canvas.draw_idle()

    def show_batch_with_predictions(self):
        self.indices = self.get_next_batch_indices()
        if len(self.indices) == 0:
            print("All crops labeled.")
            return

        self.labels = {}
        self.rects = {}

        self.fig, self.axes = plt.subplots(5, 5, figsize=(10, 10))
        self.fig.suptitle(f"{self.replica_name} | Left=SPOT, Right=NOT")
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)

        for i, idx in enumerate(self.indices):
            crop = self.crops[idx]
            ax = self.axes.flat[i]
            ax.imshow(crop, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            pred_str = self.meta.at[idx, 'rf_prediction'] if 'rf_prediction' in self.meta.columns else "?"
            ax.set_title(f"Idx {idx}\nPRED: {pred_str}")

        for j in range(len(self.indices), 25):
            self.axes.flat[j].axis('off')

        plt.tight_layout()
        plt.show()

    def save_current_labels(self):
        labeled_count = 0
        for plot_idx in range(len(self.indices)):
            crop_idx = self.indices[plot_idx]
            if crop_idx in self.labels:
                self.meta.at[crop_idx, "manual_label"] = self.labels[crop_idx]
                labeled_count += 1

        self.meta.to_csv(self.output_path, index=False)
        print(f"Saved {labeled_count} labels to {self.output_path}")

    def get_feature_matrix(self, df):
        return df.drop(columns=['manual_label', 'rf_prediction'], errors='ignore').select_dtypes(include=[np.number])

    def apply_predictions(self, clf):
        if clf is None:
            return

        unlabeled_df = self.meta[self.meta['manual_label'].isna()]
        indices = unlabeled_df.index[:self.batch_size]
        X_pred = self.get_feature_matrix(unlabeled_df.loc[indices])
        preds = clf.predict(X_pred)
        self.meta.loc[indices, 'rf_prediction'] = preds

class ClassifierLoop:
    def __init__(self, directory):
        self.directory = directory
        self.pairs = self.find_crop_metadata_pairs()
        self.global_meta = self.load_all_metadata()

    def find_crop_metadata_pairs(self):
        csv_files = sorted(glob.glob(os.path.join(self.directory, '*_all_crop_metadata.csv')))
        pairs = []
        for csv_path in csv_files:
            base = csv_path.replace('_all_crop_metadata.csv', '')
            npy_path = base + '_all_crops.npy'
            if os.path.exists(npy_path):
                pairs.append((npy_path, csv_path))
            else:
                print(f"Missing .npy for: {csv_path}")
        return pairs

    def load_all_metadata(self):
        dfs = []
        for _, csv_path in self.pairs:
            df = pd.read_csv(csv_path)
            df['source_file'] = csv_path
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def train_global_random_forest(self):
        labeled_df = self.global_meta[self.global_meta['manual_label'].notna()]
        if labeled_df.empty:
            print("No labeled data available to train.")
            return None
        features = labeled_df.drop(columns=['manual_label', 'rf_prediction', 'source_file'], errors='ignore')
        X = features.select_dtypes(include=[np.number])
        y = labeled_df['manual_label']
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        print("Trained global Random Forest on", len(X), "samples")
        return clf

    def run(self):
        for i, (crops_path, meta_path) in enumerate(self.pairs):
            print(f"\nReplica {i+1}/{len(self.pairs)}: {os.path.basename(meta_path)}")
            replica_df = pd.read_csv(meta_path)
            labeler = ManualLabeler(crops_path, meta_path, replica_df)

            while True:
                clf = self.train_global_random_forest()
                labeler.apply_predictions(clf)
                labeler.show_batch_with_predictions()

                input("Done labeling this batch? Press ENTER to save and continue...")
                labeler.save_current_labels()

                # Update the global meta with new labels
                updated_df = pd.read_csv(meta_path)
                updated_df['source_file'] = meta_path
                self.global_meta = pd.concat(
                    [df for df in self.global_meta[self.global_meta['source_file'] != meta_path]] + [updated_df],
                    ignore_index=True
                )

                remaining = labeler.meta['manual_label'].isna().sum()
                print(f"Remaining unlabeled crops: {remaining}")
                if remaining == 0:
                    print("All crops labeled for this replica!")
                    break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run interactive spot classifier loop")
    parser.add_argument('--dir', required=True, help="Path to directory with .npy and .csv files")
    args = parser.parse_args()

    loop = ClassifierLoop(args.dir)
    loop.run()
