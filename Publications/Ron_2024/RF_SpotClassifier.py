import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss

class ManualLabeler:
    def __init__(self, crops_path, meta_path, meta_df, output_path=None, model=None, threshold=0.5):
        self.crops = np.load(crops_path)
        self.meta = meta_df.copy()
        self.meta_path = meta_path
        self.replica_name = os.path.basename(meta_path).replace("_all_crop_metadata.csv", "")
        self.output_path = output_path or meta_path

        # ensure manual_label column exists
        if "manual_label" not in self.meta.columns:
            self.meta["manual_label"] = np.nan

        # decision threshold for probability→class
        self.threshold = threshold

        # load pre-trained model if you really need it (we don’t use it below)
        self.model = None
        if model and os.path.exists(model):
            self.model = joblib.load(model)
            print(f"Loaded pre-trained model from: {model}")

        self.batch_size = 25
        self.fig = None
        self.axes = None
        self.rects = {}
        self.labels = {}
        self.indices = []

    def get_next_batch_indices(self):
        unlabeled = self.meta[self.meta["manual_label"].isna()].index.tolist()
        if not unlabeled:
            return []
        n = min(self.batch_size, len(unlabeled))
        return list(np.random.choice(unlabeled, size=n, replace=False))

    def onclick(self, event):
        for i, ax in enumerate(self.axes.flat):
            if event.inaxes == ax:
                idx = self.indices[i]
                self.labels[idx] = 1 if event.button == 1 else 0
                self.update_border(i, self.labels[idx])
                break

    def update_border(self, plot_idx, label):
        ax = self.axes.flat[plot_idx]
        # remove old rectangle
        if self.rects.get(plot_idx):
            self.rects[plot_idx].remove()

        # green=spot, red=not-spot
        color = "green" if label == 1 else "red"
        self.rects[plot_idx] = ax.add_patch(
            Rectangle((0,0),1,1, transform=ax.transAxes,
                      fill=False, edgecolor=color, linewidth=3)
        )

        idx = self.indices[plot_idx]
        pred = self.meta.at[idx, 'rf_prediction'] if 'rf_prediction' in self.meta.columns else "?"
        proba = self.meta.at[idx, 'rf_probability'] if 'rf_probability' in self.meta.columns else None
        try:
            pstr = f"{float(proba):.2f}"
        except (TypeError, ValueError):
            pstr = "?"
        label_str = "[SPOT]" if label == 1 else "[NOT]"

        ax.set_title(f"Idx {idx}\nPRED: {pred} (p={pstr}) → {label_str}")
        self.fig.canvas.draw_idle()

    def show_batch_with_predictions(self):
        self.indices = self.get_next_batch_indices()
        if not self.indices:
            print("All crops labeled.")
            return

        self.labels.clear()
        self.rects.clear()

        self.fig, self.axes = plt.subplots(5,5, figsize=(10,10))
        self.fig.suptitle(f"{self.replica_name} | Left-click = SPOT, Right-click = NOT")
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)

        for i, idx in enumerate(self.indices):
            crop = self.crops[idx]
            ax = self.axes.flat[i]
            ax.imshow(crop, cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])

            pred = self.meta.at[idx, 'rf_prediction'] if 'rf_prediction' in self.meta.columns else "?"
            proba = self.meta.at[idx, 'rf_probability'] if 'rf_probability' in self.meta.columns else None
            try:
                pstr = f"{float(proba):.2f}"
            except (TypeError, ValueError):
                pstr = "?"
            ax.set_title(f"Idx {idx}\nPRED: {pred} (p={pstr})")

        # hide unused axes
        for j in range(len(self.indices), 25):
            self.axes.flat[j].axis('off')

        plt.tight_layout()
        plt.show()

    def save_current_labels(self):
        count = 0
        for plot_idx in range(len(self.indices)):
            crop_idx = self.indices[plot_idx]
            if crop_idx in self.labels:
                self.meta.at[crop_idx, "manual_label"] = self.labels[crop_idx]
                count += 1

        # save the full metadata (including rf_prediction & rf_probability)
        self.meta.to_csv(self.output_path, index=False)
        print(f"Saved {count} labels to {self.output_path}")

    def get_feature_matrix(self, df):
        """
        Drop manual_label, rf_prediction, rf_probability so that the
        classifier is always trained on the original numeric features only.
        """
        return (
            df
            .drop(columns=['manual_label','rf_prediction','rf_probability'], errors='ignore')
            .select_dtypes(include=[np.number])
        )

    def apply_predictions(self, clf):
        if clf is None or len(clf.classes_) < 2:
            print("Skipping predictions (no sufficient classes).")
            return

        unlabeled = self.meta[self.meta['manual_label'].isna()]
        indices = unlabeled.index[:self.batch_size]
        if len(indices)==0:
            print("No unlabeled crops left.")
            return

        X = self.get_feature_matrix(unlabeled.loc[indices])
        if X.empty:
            print("No numeric features for prediction.")
            return

        probs = clf.predict_proba(X)[:,1]
        preds = (probs >= self.threshold).astype(int)
        self.meta.loc[indices, 'rf_prediction'] = preds
        self.meta.loc[indices, 'rf_probability'] = probs

    def show_training_progress(self, clf):
        labeled = self.meta[self.meta['manual_label'].notna()]
        if labeled.empty or clf is None:
            print("No training progress to show.")
            return

        y_true = labeled['manual_label'].astype(int)
        classes = sorted(y_true.unique())
        if len(classes) < 2:
            print(f"Only one class ({classes[0]}) labeled so far; skipping log loss.")
            return

        X = self.get_feature_matrix(labeled)
        y_proba = clf.predict_proba(X)
        loss = log_loss(y_true, y_proba, labels=classes)
        print(f"Log loss on current labeled data: {loss:.4f}")

        plt.figure(figsize=(8,4))
        for lbl in [0,1]:
            plt.hist(y_proba[y_true==lbl,1], bins=20, alpha=0.6,
                     label=f"Manual label = {lbl}")
        plt.xlabel("Predicted Probability (class 1)")
        plt.ylabel("Count")
        plt.title(f"Prediction vs Manual Labels\nLog Loss: {loss:.4f}")
        plt.legend()
        plt.tight_layout()
        plt.show()

class ClassifierLoop:
    def __init__(self, directory):
        self.directory = directory
        self.pairs = self.find_crop_metadata_pairs()
        self.global_meta = self.load_all_metadata()

    def find_crop_metadata_pairs(self):
        csvs = sorted(glob.glob(os.path.join(self.directory, '*_all_crop_metadata.csv')))
        pairs = []
        for csv in csvs:
            npy = csv.replace('_all_crop_metadata.csv', '_all_crops.npy')
            if os.path.exists(npy):
                pairs.append((npy, csv))
            else:
                print(f"Missing .npy for: {csv}")
        return pairs

    def load_all_metadata(self):
        dfs = []
        for _, csv in self.pairs:
            df = pd.read_csv(csv)
            df['source_file'] = csv
            dfs.append(df)
        meta = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(columns=['source_file'])
        if 'manual_label' not in meta.columns:
            meta['manual_label'] = np.nan
        return meta

    def train_global_random_forest(self):
        """
        Train on ALL labeled data across replicas, dropping
        manual_label, source_file, rf_prediction, rf_probability.
        """
        labeled = self.global_meta[self.global_meta['manual_label'].notna()]
        if labeled.empty:
            print("No labeled data available to train.")
            return None

        features = labeled.drop(columns=['manual_label','source_file','rf_prediction','rf_probability'],
                                 errors='ignore')
        X = features.select_dtypes(include=[np.number])
        y = labeled['manual_label'].astype(int)

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X, y)
        print("Trained global RF on", len(X), "samples")

        joblib.dump(clf, os.path.join(self.directory, "global_random_forest.pkl"))
        return clf

    def run(self, model_path=None):
        for i, (crops, meta) in enumerate(self.pairs):
            print(f"\nReplica {i+1}/{len(self.pairs)}: {os.path.basename(meta)}")
            df = pd.read_csv(meta)
            if 'manual_label' not in df.columns:
                df['manual_label'] = np.nan

            labeler = ManualLabeler(crops, meta, df, model=model_path)

            while True:
                clf = self.train_global_random_forest()
                labeler.apply_predictions(clf)
                labeler.show_batch_with_predictions()
                labeler.show_training_progress(clf)

                ans = input(
                    "Done labeling this batch? ENTER to save & continue, "
                    "or 'N'+ENTER to skip rest of this replica: "
                ).strip().lower()

                labeler.save_current_labels()

                updated = pd.read_csv(meta)
                updated['source_file'] = meta
                if 'manual_label' not in updated.columns:
                    updated['manual_label'] = np.nan

                self.global_meta = pd.concat(
                    [self.global_meta[self.global_meta['source_file'] != meta],
                     updated],
                    ignore_index=True
                )

                if ans.startswith('n'):
                    print("Skipping remaining crops for this replica.\n")
                    break

                rem = labeler.meta['manual_label'].isna().sum()
                print(f"Remaining unlabeled crops: {rem}")
                if rem == 0:
                    print("All crops labeled for this replica.\n")
                    break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Run interactive spot classifier loop")
    parser.add_argument('--dir', required=True,
                        help="Path to directory with .npy and .csv files")
    args = parser.parse_args()

    loop = ClassifierLoop(args.dir)
    loop.run(model_path=os.path.join(args.dir, "global_random_forest.pkl"))