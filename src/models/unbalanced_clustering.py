import numpy as np
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
import logging
import matplotlib.colors as colors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from collections import Counter
from typing import Any


class UnbalancedClustering():
    """
    Apply XGBoost clustering to an unbalanced dataset, train and evaluate, and infer predictions for new samples.
    This class is written in a general sense and could be applied to any unbalanced binary dataset, not just the transactional
    dataset used for development. This allows for scalability and repeatability, especially if clients have different
    dataset aggregation methods.
    """
    def __init__(
            self,
            X: np.ndarray,
            y: np.ndarray,
            n_splits: int,
            features: list[str],
            show_fig: bool=True
            ) -> None:
        """
        Initialize values for Unbalanced Clustering class.

        Parameters
        ----------
        X : np.ndarray
            independent parameters, df[columns].values

        y : np.ndarray
            target parameter, df[target].values

        n_splits : int
            the number of folds to use with Stratified K-Fold cross-validation

        features : list of str
            the names of the X columns

        show_fig : bool
            if show_fig is true, model progress will be plotted, shown, and returned as objects

        Returns
        -------
        None
        """
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.features = features
        self.show_fig = show_fig

    def set_params(self) -> dict:
        """
        Set the parameters to be used for XGBoost, including calculating the class imbalance ratio.

        Returns
        -------
        params : dict
            XGBoost parameters
        """
        # Determine the ratio of the classes
        _, counts = np.unique(self.y, return_counts=True)
        class_ratio = counts[0] / counts[1]
        
        # Set Training Parameters
        params = {
            "objective": "binary:logistic",  # just has 2 classes, 0/1
            "eval_metric": "aucpr",  # Focus on the area under the Precision-Recall curve, how well do we find the needle in the haystack?
            "scale_pos_weight": class_ratio,  # i.e., (Total Legit / Total Fraud)
            "max_depth": 4,
            "eta": 0.1,  # learning rate
            "subsample": 0.8,
            "colsample_bytree": 1.0
        }

        return params

    def train_predict_evaluate_split(
            self,
            X: np.ndarray,
            y: np.ndarray,
            params: dict,
            train_index: np.ndarray[int],
            test_index: np.ndarray[int]
            ) -> tuple[np.ndarray, np.ndarray, float, xgb.Booster]:
        """
        Using the provided indices, split X and y into train and test sets, convert to DMatrices (for native XGBoost),
        apply training parameters, train with XGBoost, extract precision, recall, and area under the precision-recall curve.

        Parameters
        ----------
        X : np.ndarray
            independent parameters, df[columns].values

        y : np.ndarray
            target parameter, df[target].values

        params : dict
            XGBoost parameters

        train_index : np.ndarray of int
            array of indices to be used in training

        test_index : np.ndarray of int
            array of indices to be used in testing

        Returns
        -------
        precision : np.ndarray
            From sklearn.metrics.precision_recall_curve documentation:
            Precision values such that element i is the precision of predictions with score >= thresholds[i]
            and the last element is 1.

        recall : np.ndarray
            From sklearn.metrics.precision_recall_curve documentation:
            Decreasing recall values such that element i is the recall of predictions with score >= thresholds[i]
            and the last element is 0.
        
        pr_auc : float
            Area Under the Curve.

        model : xgboost.Booster
            Trained XGBoost model
        """
        logging.debug("Starting Training for current split")

        # split into train and test splits
        self.X_train, self.X_test = X[train_index], X[test_index]
        self.y_train, self.y_test = y[train_index], y[test_index]

        # Convert into DMatrix to save on memory, pre-calculate quantiles for split-points, and handle missing values
        dtrain = xgb.DMatrix(self.X_train, label=self.y_train)
        dtest = xgb.DMatrix(self.X_test, label=self.y_test)

        # Train using native XGBoost
        model = xgb.train(params, dtrain, num_boost_round=1000)
        
        # Predict and evaluate
        self.y_pred = model.predict(dtest)
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred)
        pr_auc = auc(recall, precision)

        logging.debug("Training complete for current split")

        return precision, recall, pr_auc, model

    def calculate_metrics(self, y_true: list[np.ndarray], y_pred: list[np.ndarray]) -> dict[str, Any]:
        """ 
        Calculate the metrics of the trained XGBoost model across all stratified K-folds.
        
        Parameters
        ----------
        y_true : list of np.ndarray
            the actual values of the target
        
        y_pred : list of np.ndarray
            the probability values of the target

        Returns
        -------
        metrics : dict of str | Any
            the metrics for the trained XGBoost model across all stratified K-folds
        """
        # Calculate final precision, recall, threshold, and f1 scores
        final_precision, final_recall, final_threshold = precision_recall_curve(y_true, y_pred)
        f1_scores = (2 * final_precision * final_recall) / (final_precision + final_recall + 1e-10)
        best_idx = np.argmax(f1_scores)

        # Populate metrics into output dict to cut down on returned parameters while maintaining clarity
        metrics = {
            "precision": final_precision,
            "recall": final_recall,
            "threshold": final_threshold,
            "pr_auc": auc(final_recall, final_precision),
            "f1_scores": f1_scores,
            "best_threshold": final_threshold[best_idx],
            "best_f1_score": f1_scores[best_idx],
            "precision_at_best_threshold": final_precision[best_idx],
            "recall_at_best_threshold": final_recall[best_idx]
        }

        # Output logs
        logging.debug(f"*** Global Performance ***")
        logging.debug(f"Optimal Threshold: {metrics['best_threshold']:.4f}")
        logging.debug(f"Best F1-Score: {metrics['best_f1_score']:.4f}")
        logging.debug(f"Precision at this point: {metrics['precision_at_best_threshold']:.4f}")
        logging.debug(f"Recall at this point: {metrics['recall_at_best_threshold']:.4f}")

        return metrics

    def get_feature_importance(
            self,
            importances: list[dict[str, float | list[float]]],
            n_splits: int,
            features: list[str]
            ) -> list[tuple[str, float]]:
        """ 
        Get the feature importance for the trained XGBoost model across all stratified K-folds. The average of all
        the splits is returned.

        Parameters
        ----------
        importances : list of dict
            list of importance scores for each stratified K-Fold

        n_splits : int
            the number of folds to use with Stratified K-Fold cross-validation

        features : list of str
            the names of the X columns

        Returns
        -------
        sorted_importances : list of tuple
            (feature, score)
        """
        # Combine and average the feature importance scores
        total_importance = Counter()
        for score in importances:
            total_importance.update(score)

        # Divide by number of folds to get the average gain
        avg_importance = {k: (v / n_splits) for k, v in total_importance.items()}

        # Map back to feature names
        final_importance = {}
        for k, v in avg_importance.items():
            idx = int(k.replace('f', ''))  # Convert 'f0' -> 0
            final_importance[features[idx]] = v

        sorted_importances = sorted(final_importance.items(), key=lambda x: x[1], reverse=True)
        
        logging.debug("*** Average Top 5 Features ***")
        for name, score in sorted_importances[:5]:
            logging.debug(f"{name}: {score:.2f}")

        return sorted_importances
        
    def train_imbalanced_model(
            self
            ) -> tuple[
                list[np.ndarray],
                list[np.ndarray],
                dict[str, Any],
                list[tuple[str, float]],
                matplotlib.figure.Figure
                ]:
        """
        Set up the parameters for the XGBoost model, including the class imbalance ratio, implement a stratified K-Fold split,
        then for each split train the model, evaluate, calculate metrics, get feature importance, and plot progress if requested.

        Save trained model for inference.

        Returns
        -------
        all_y_true : list of np.ndarray
            the actual values of the target
        
        all_y_pred : list of np.ndarray
            the probability values of the target

        metrics : dict of str | Any
            the metrics for the trained XGBoost model across all stratified K-folds

        sorted_importances : list of tuple
            (feature, score)

        fig : matplotlib.figure.Figure
            precision-recall curve

        """
        # Set the training parameters
        params = self.set_params()

        # Initialize KFold
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=45)

        fig = None
        if self.show_fig is True:
            plt.figure(figsize=(10, 3))

        # Set up data containers    
        precision, recall, pr_auc = {}, {}, {}
        importances, all_y_true, all_y_pred = [], [], []

        # Loop through splits, train, predict, evaluate
        for fold, (train_index, test_index) in enumerate(skf.split(self.X, self.y)):
            logging.debug(f"Beginning Stratified K-Fold Split {fold}")
            precision[fold], recall[fold], pr_auc[fold], model = self.train_predict_evaluate_split(
                self.X, self.y, params, train_index, test_index
                )

            # Update overall lists for test set and predictions
            all_y_true.extend(self.y_test)
            all_y_pred.extend(self.y_pred)

            # Get feature importances
            fold_importance = model.get_score(importance_type='gain')
            importances.append(fold_importance)

            if self.show_fig is True:
                # plot current split
                plt.plot(
                    recall[fold],
                    precision[fold],
                    color='grey',
                    alpha=0.3,
                    label=f'Fold {fold} PR-AUC: {pr_auc[fold]:.3f}'
                    )
        
        mean_pr_auc = np.mean([val for val in pr_auc.values()])
        logging.info(f"Average PR-AUC across {self.n_splits} folds: {mean_pr_auc:.4f}")

        # calculate final metrics for overall predictions
        metrics = self.calculate_metrics(all_y_true, all_y_pred)

        # get feature importance
        sorted_importances = self.get_feature_importance(importances, self.n_splits, self.features)

        # Save model
        model.save_model('../models/resources/credit_fraud_model.json')

        if self.show_fig is True:
            plt.plot(metrics["recall"], metrics["precision"], color='#2ecc71', lw=3, label=f'Mean PR (AUC = {metrics["pr_auc"]:.3f})')
            plt.plot(metrics["recall_at_best_threshold"], metrics["precision_at_best_threshold"], 'o', color="#0072B2", ms=7,
                    label=f"Best Threshold = {metrics['best_threshold']:.4f}")

            # Add a baseline for a "no-skill" model (ratio of fraud in dataset)
            baseline = len(self.y_test[self.y_test==1]) / len(self.y_test)
            plt.axhline(y=baseline, color='darkorange', linestyle='--', label=f'Baseline ({baseline:.4f})')

            plt.xlabel('Recall (Catching Problems)')
            plt.ylabel('Precision (Avoiding False Alarms)')
            plt.title('Multi-Fold Performance: Heavily Imbalanced Dataset')
            plt.legend(loc='lower left', fontsize='small', ncol=2)
            plt.grid(alpha=0.2)
            plt.show()

        return all_y_true, all_y_pred, metrics, sorted_importances, fig

    def apply_threshold(
            self,
            y_test: list[np.ndarray],
            y_pred: list[np.ndarray],
            optimal_threshold: int
            ) -> tuple[np.ndarray, np.ndarray, matplotlib.figure.Figure]:
        """ 
        Compare the output probabilities to the optimal threshold to assign predicted classes.

        Parameters
        ----------
        y_test : list of np.ndarray
            the actual values of the target
        
        y_pred : list of np.ndarray
            the probability values of the target

        optimal_threshold : float
            the best threshold for choosing between target classes
            (e.g., probability greater than threshold = 0.5 is class 1)

        Returns
        -------
        y_final_predictions : np.ndarray
            classes of predictions

        cm : np.ndarray
            confusion matrix
            [[TP, FN],
             [FP, TN]]
        
        fig : matplotlib.figure.Figure
            confusion matrix
        """
        # Apply optimal threshold to predictions
        y_final_predictions = (y_pred >= optimal_threshold).astype(int)

        # Generate the matrix
        cm = confusion_matrix(y_test, y_final_predictions)
        logging.debug(f"Annoyed clients = {cm[0][1]/(cm[0][0]+cm[0][1]):.3e}, Missed fraud = {cm[1][0]/(cm[1][0]+cm[1][1]):.3e}")

        fig = None
        if self.show_fig is True:
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens, norm=colors.LogNorm(vmin=max(1, cm.min()), vmax=cm.max()))
            ax.figure.colorbar(im, ax=ax)

            # Setup labels
            classes = ['Legit', 'Fraud']
            tick_marks = np.arange(len(classes))
            ax.set_xticks(tick_marks)
            ax.set_xticklabels(classes)
            ax.set_yticks(tick_marks)
            ax.set_yticklabels(classes)

            # Adjust text color logic for log scale
            thresh = np.exp(np.log(cm.max()) / 2.) 
            for i in range(cm.shape[0]):    # Add [0] here
                for j in range(cm.shape[1]): # Add [1] here
                    ax.text(j, i, format(cm[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if cm[i, j] > thresh else "black")

            ax.set_title('Log-Scaled Confusion Matrix')

            ax.set_ylabel('Actual Label')
            ax.set_xlabel('Predicted Label')
            plt.tight_layout()
            plt.show()

        return y_final_predictions, cm, fig
    
    def run_training(self) -> tuple[
        np.ndarray,
        np.ndarray,
        dict[str, Any],
        list[tuple[str, float]],
        matplotlib.figure.Figure,
        matplotlib.figure.Figure
        ]:
        """ 
        Returns
        -------
        y_final_predictions : np.ndarray
            classes of predictions

        cm : np.ndarray
            confusion matrix
            [[TP, FN],
             [FP, TN]]

        metrics : dict of str | Any
            the metrics for the trained XGBoost model across all stratified K-folds

        sorted_importances : list of tuple
            (feature, score)
        
        fig1 : matplotlib.figure.Figure
            precision-recall curve

        fig2 : matplotlib.figure.Figure
            confusion matrix
        """
        all_y_true, all_y_pred, metrics, sorted_importances, fig1 = self.train_imbalanced_model()
        y_final_predictions, cm, fig2 = self.apply_threshold(all_y_true, all_y_pred, metrics["best_threshold"])
        return y_final_predictions, cm, metrics, sorted_importances, fig1, fig2
