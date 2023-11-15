from metaflow import FlowSpec, step, batch, pypi_base, card, current
from metaflow.cards import Image

# user packages
from dependencies import *
from ops import ModelStore
from fraud_detection_logic import FeatureEngineering, ModelTraining

@pypi_base(
    python=python_version,
    packages={
        **pypi_common_pkgs,
        **pypi_feature_eng_pkgs,
        # **pypi_tf_pkg,
        **pypi_xgb_pkg,
    },
)
class FraudClassifierTreeSelection(
    FlowSpec, 
    FeatureEngineering, 
    ModelTraining, 
    ModelStore # introduces required param "model-repo" expecting s3 uri the flow's task execution role can write to
):
    _plot_learning_curves = True

    @step
    def start(self):
        self.next(self.preprocess)

    @batch(cpu=1, memory=8000)
    @card
    @step
    def preprocess(self):
        self.compute_features()
        self.setup_model_grid(
            model_list=["Random Forest"]
        )
        self.next(self.train, foreach="model_grid")

    @batch(cpu=4, memory=16000)
    @card
    @step
    def train(self):
        self.model_name, self.model_grid = self.input
        self.best_model = self.smote_pipe(
            self.model_grid, self.X_train_full, self.y_train_full
        )
        self.next(self.eval)

    @batch(cpu=1, memory=8000)
    @card
    @step
    def eval(self, inputs):

        # propagate data artifacts
        self.X_train_full = inputs[0].X_train_full
        self.X_test_full = inputs[0].X_test_full
        self.y_train_full = inputs[0].y_train_full
        self.y_test_full = inputs[0].y_test_full

        # score trained models
        from my_fraud_detection_logic import score_trained_model
        import pandas as pd

        best_score = -1
        self.best_model = None
        self.best_model_type = None
        scores = []
        best_models = []
        for input in inputs:
            scores.append(
                {
                    "model name": input.model_name,
                    **score_trained_model(
                        input.best_model, self.X_test_full, self.y_test_full
                    ),
                }
            )
            best_models.append((input.best_model, input.model_name))
            if scores[-1]["auc"] > best_score:
                best_score = scores[-1]["auc"]
                self.best_model = input.best_model
                self.best_model_type = input.model_name
        self.scores = pd.DataFrame(scores)

        # push best model - function defined in 
        self.store_sklearn_estimator(model=self.best_model)

        # plot learning curves
        if self._plot_learning_curves:
            import matplotlib.pyplot as plt

            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
                2, 2, figsize=(18, 10), sharey=True
            )
            axs = [ax1, ax2, ax3, ax4]
            self.plot_learning_curves(
                best_models, axs, self.X_train_full, self.y_train_full
            )
            fig.tight_layout()
            current.card.append(Image.from_matplotlib(fig))

        self.next(self.end)

    @step
    def end(self):
        print(
            f"""
        Access evaluation results:

        from metaflow import Flow
        f = Flow('{current.flow_name}')
        r = f.latest_successful_run
        scores = r.data.scores
        """
        )

if __name__ == "__main__":
    FraudClassifierTreeSelection()