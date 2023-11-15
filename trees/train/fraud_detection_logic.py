# Original source: https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets

import logging
import os
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Data manipulation
import numpy as np
import pandas as pd
import time

# Data viz
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from metaflow.cards import Image
from metaflow import current, card

# Featurize & Data split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

# Classifier Libraries
# import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
import collections

# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from imblearn.pipeline import make_pipeline as imbalanced_make_pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from imblearn.metrics import classification_report_imbalanced
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    accuracy_score,
    classification_report,
)
from collections import Counter
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
)

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*is_sparse is deprecated and will be removed in a future version.*",
)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*is_categorical is deprecated and will be removed in a future version.*",
)

BACKGROUND = "#F4EBE6"
GREEN = "#37795D"
PURPLE = "#5460C0"
colors = [GREEN, PURPLE]

custom_params = {
    "axes.spines.right": False,
    "axes.spines.top": False,
    "axes.facecolor": BACKGROUND,
    "figure.facecolor": BACKGROUND,
    "figure.figsize": (8, 8),
}
sns.set_theme(style="ticks", rc=custom_params)

# cmap = LinearSegmentedColormap.from_list("Custom", colors, N=20)
cmap = "viridis"

default_model_grid = {
    "Logistic Regression": GridSearchCV(
        LogisticRegression(),
        {"C": [0.05, 0.1, 1], "max_iter": [300, 500, 700]},
        verbose=2,
    ),
    "K Nearest Neighbors": GridSearchCV(
        KNeighborsClassifier(),
        {
            "n_neighbors": list(range(2, 5, 1)),
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
        },
        verbose=2,
    ),
    "SVM": GridSearchCV(
        SVC(),
        {
            "C": [0.5],  # , 0.7, 0.9, 1],
            "kernel": ["rbf"],  # , 'poly', 'sigmoid', 'linear']
        },
        verbose=2,
    ),
    "Decision Tree": GridSearchCV(
        DecisionTreeClassifier(),
        {
            "criterion": ["gini"], #, "entropy"],
            "max_depth": [2,4,6],
            # "min_samples_leaf": list(range(5, 7, 1)),
        },
        verbose=2,
    ),
    "Random Forest": GridSearchCV(
        RandomForestClassifier(),
        {
            "criterion": ["gini"], #, "entropy"],
            "max_depth": [2],
            "n_estimators": [50],  # , 100, 150],
        },
    ),
    "Extra Trees": GridSearchCV(
        ExtraTreeClassifier(),
        {
            "criterion": ["gini"], #, "entropy"],
            "max_depth": [2,4,6],
        },
        verbose=2,
    ),
    "Gradient Boost": GridSearchCV(
        GradientBoostingClassifier(),
        {
            "criterion": ["squared_error"], #, "friedman_mse"],
            "max_depth": [2,4,6],
        },
        verbose=2,
    ),
    # "XGBoost": GridSearchCV(
    #     XGBClassifier(),
    #     {
    #         "max_depth": [2],  # , 4, 6],
    #         "n_estimators": [50],  # , 100, 150],
    #         "learning_rate": [0.01],  # , 0.1, 0.2]
    #     },
    #     verbose=2,
    # ),
}


class FeatureEngineering(object):
    filename = "creditcard.csv"
    bucket = "s3://outerbounds-datasets/credit-card-fraud-detection"

    def download_data_to_instance(self, out_path):
        from metaflow import S3
        import os

        with S3() as s3:
            obj = s3.get(os.path.join(self.bucket, self.filename))
            os.rename(obj.path, out_path)
        assert os.path.exists(out_path)

    def compute_features(self, out_path=None):
        if not out_path:
            out_path = os.path.abspath(self.filename)

        # move data from s3 to instance
        from my_fraud_detection_logic import load_data

        self.download_data_to_instance(out_path=out_path)
        self.df = load_data(data_path=self.filename)

        # plot original class distribution
        from my_fraud_detection_logic import plot_classes
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        plot_classes(self.df, ax=ax)
        current.card.append(Image.from_matplotlib(fig))

        # split and store full data
        from my_fraud_detection_logic import train_test_split_full_data

        (
            self.X_train_full,
            self.X_test_full,
            self.y_train_full,
            self.y_test_full,
        ) = train_test_split_full_data(self.df)

        # scale features
        from my_fraud_detection_logic import scale_features

        self.df_scaled = scale_features(self.df)

        # random undersample
        # from my_fraud_detection_logic import random_undersample
        # self.df_undersample = random_undersample(self.df_scaled)

        # plot correlation matrices
        # from my_fraud_detection_logic import plot_correlation_matrices
        # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18,10))
        # fig.tight_layout()
        # plot_correlation_matrices(self.df, self.df_undersample, ax1, ax2)
        # current.card.append(Image.from_matplotlib(fig))

        # remove outliers
        from my_fraud_detection_logic import remove_outliers_from_fraud_transactions

        self.df_filtered = remove_outliers_from_fraud_transactions(
            self.df_scaled, ["V10", "V12", "V14"]
        )

        # split data
        from my_fraud_detection_logic import split_data

        self.X_train, self.X_test, self.y_train, self.y_test = split_data(
            self.df_filtered
        )


class ModelTraining(object):
    def __init__(self):
        pass

    def setup_model_grid(self, model_list=[]):
        if model_list == []:
            model_list = default_model_grid.keys()

        self.model_grid = []  # passed as foreach split for modeling phase
        for model_name in model_list:
            if model_name not in default_model_grid:
                print(f"Invalid model name: {model_name}. Skipping...")
            else:
                self.model_grid.append((model_name, default_model_grid[model_name]))
        logging.info(f"Model grid: {self.model_grid}")

    def smote_pipe(self, model_grid, X_train, y_train):
        pipeline = imbalanced_make_pipeline(
            SMOTE(sampling_strategy="minority"), model_grid
        )
        logging.info(f"Fitting model grid with params {self.model_grid}...")
        pipeline.fit(X_train, y_train)
        logging.info(f"Fitting complete.")
        return model_grid.best_estimator_

    def plot_learning_curves(
        self,
        estimators,
        axs,
        X,
        y,
        ylim=None,
        cv=None,
        n_jobs=1,
        train_sizes=np.linspace(0.1, 1.0, 5),
    ):
        if ylim is not None:
            plt.ylim(*ylim)

        for (estimator, estimator_name), ax in zip(estimators, axs):
            train_sizes, train_scores, test_scores = learning_curve(
                estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
            )
            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            ax.fill_between(
                train_sizes,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.1,
                color=GREEN,
            )
            ax.fill_between(
                train_sizes,
                test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,
                alpha=0.1,
                color=PURPLE,
            )
            ax.plot(
                train_sizes,
                train_scores_mean,
                "o-",
                color=GREEN,
                label="Training score",
            )
            ax.plot(
                train_sizes,
                test_scores_mean,
                "o-",
                color=PURPLE,
                label="Cross-validation score",
            )
            ax.set_title(f"{estimator_name} learning curve", fontsize=14)
            ax.set_xlabel("Training size")
            ax.set_ylabel("Score")
            ax.grid(True)
            ax.legend(loc="best")


def load_data(data_path="creditcard.csv"):
    df = pd.read_csv(data_path)
    return df


def plot_classes(df, ax=None):
    sns.countplot(data=df, x="Class", hue="Class", ax=ax)
    ax.set_title("Class Distributions \n (0: No Fraud || 1: Fraud)", fontsize=14)


def scale_features(df, scaler="robust"):
    logging.info("Scaling features")

    if scaler == "standard":
        scaler = StandardScaler()
    elif scaler == "robust":
        scaler = RobustScaler()
    else:
        raise Exception("Invalid scaler")

    df["scaled_amount"] = scaler.fit_transform(df["Amount"].values.reshape(-1, 1))
    df["scaled_time"] = scaler.fit_transform(df["Time"].values.reshape(-1, 1))
    df.drop(["Time", "Amount"], axis=1, inplace=True)

    scaled_amount = df["scaled_amount"]
    scaled_time = df["scaled_time"]

    df.drop(["scaled_amount", "scaled_time"], axis=1, inplace=True)
    df.insert(0, "scaled_amount", scaled_amount)
    df.insert(1, "scaled_time", scaled_time)

    return df


def random_undersample(df):
    logging.info("Random undersampling")

    # shuffle
    df = df.sample(frac=1)

    # amount of fraud classes 492 rows.
    fraud_df = df.loc[df["Class"] == 1]
    non_fraud_df = df.loc[df["Class"] == 0][: fraud_df.shape[0]]
    evenly_distributed_df = pd.concat([fraud_df, non_fraud_df])

    # shuffle
    evenly_distributed_df = evenly_distributed_df.sample(frac=1, random_state=42)

    return evenly_distributed_df


def plot_correlation_matrices(original_df, resampled_df, ax1=None, ax2=None):
    logging.info("Plotting correlation matrices")

    corr = original_df.corr()
    sns.heatmap(corr, cmap=cmap, annot_kws={"size": 14}, ax=ax1)
    ax1.set_title(
        "Imbalanced Correlation Matrix \n (don't use for reference)", fontsize=14
    )

    sub_sample_corr = resampled_df.corr()
    sns.heatmap(sub_sample_corr, cmap=cmap, annot_kws={"size": 14}, ax=ax2)
    ax2.set_title("Resampled Correlation Matrix \n (use for reference)", fontsize=14)


def remove_outliers_from_fraud_transactions(
    df=None, feat_names=["V10", "V12", "V14"], outlier_percentiles=[25, 75]
):
    original_n_rows = df.shape[0]

    for feat_name in feat_names:
        feat_fraud = df[feat_name].loc[df["Class"] == 1].values
        q1, q2 = np.percentile(feat_fraud, outlier_percentiles[0]), np.percentile(
            feat_fraud, outlier_percentiles[1]
        )
        logging.info(
            "Quartile {}: {} | Quartile {}: {}".format(
                outlier_percentiles[0], q1, outlier_percentiles[1], q2
            )
        )
        feat_iqr = q2 - q1
        logging.info("iqr: {}".format(feat_iqr))

        feat_cut_off = feat_iqr * 1.5
        feat_lower, feat_upper = q1 - feat_cut_off, q2 + feat_cut_off
        logging.info("Cut Off: {}".format(feat_cut_off))
        logging.info("{} Lower: {}".format(feat_name, feat_lower))
        logging.info("{} Upper: {}".format(feat_name, feat_upper))

        outliers = [x for x in feat_fraud if x < feat_lower or x > feat_upper]
        logging.info(
            "Feature {} Outliers for Fraud Cases: {}".format(feat_name, len(outliers))
        )
        logging.info(
            "Removing column {} outliers with values: {}".format(feat_name, outliers)
        )

        df = df.drop(
            df[(df[feat_name] > feat_upper) | (df[feat_name] < feat_lower)].index
        )

    final_n_rows = df.shape[0]
    logging.info(
        "Number of Instances Removed: {} out of {}".format(
            original_n_rows - final_n_rows, original_n_rows
        )
    )

    return df


def train_test_split_full_data(df, test_size=0.2):
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def split_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    return X_train, X_test, y_train, y_test


def train_smote_model(model_grid, model_name, X_train, y_train, n_splits=5):
    # SMOTE oversampling during cross validation
    pipeline = imbalanced_make_pipeline(SMOTE(sampling_strategy="minority"), model_grid)
    model = pipeline.fit(X_train, y_train)
    best_est = model_grid.best_estimator_

    return best_est


def score_trained_model(model, X_train, y_train):
    y_pred = model.predict(X_train)
    return {
        "accuracy": accuracy_score(y_train, y_pred),
        "recall": recall_score(y_train, y_pred),
        "precision": precision_score(y_train, y_pred),
        "f1": f1_score(y_train, y_pred),
        "auc": roc_auc_score(y_train, y_pred),
    }
