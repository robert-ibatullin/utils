import numpy as np
import pandas as pd
import shap

def shap_based_feature_selection(base_model,
                                 X: np.ndarray | pd.DataFrame,
                                 y: np.ndarray | pd.Series,
                                 n_most_important_features: int,
                                 prediction_contribution_relative_threshold: float | None = None,
                                 ) -> pd.DataFrame:
    '''
    Selects the most important features based on prediction contribution and error contribution.
    For the error contribution, see: 
        https://towardsdatascience.com/your-features-are-important-it-doesnt-mean-they-are-good-ff468ae2e3d4
    The code is taken from: 
        https://github.com/smazzanti/tds_features_important_doesnt_mean_good/blob/main/classification.ipynb
    The feature selection steps are as follows:
        1. Filter out features with positive error contribution.
        2. Sort the remaining features by prediction contribution and take top N features.
        3. If prediction_contribution_relative_threshold is not None (e.g. = 0.1), we drop features 
            having their prediction contribution < 0.1 * that of the topmost feature.
    Parameters:
        base_model: An instance of a tree-based classifier (e.g. catboost) trained on the full set of features.
        X: A numpy array or a pandas DataFrame containing the features.
        y: A numpy array or a pandas Series containing the binary target values.
        n_most_important_features: The number of most important features to select.
        prediction_contribution_relative_threshold: The minimal prediction contribution 
            relative to that of the topmost feature.
    '''

    def shap_sum2proba(shap_sum):
        """Compute sigmoid function of the Shap sum to get predicted probability."""
        return 1 / (1 + np.exp(-shap_sum))

    def individual_log_loss(y_true, y_pred, eps = 1e-15):
        """Compute log-loss for each individual of the sample."""
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    def get_preds_shaps(model, df):
        """Get predictions (predicted probabilities) and SHAP values for a dataset."""
        preds = pd.Series(model.predict_proba(df)[:,1], index=df.index)
        shap_explainer = shap.TreeExplainer(model)
        shap_expected_value = shap_explainer.expected_value[-1]
        shaps = pd.DataFrame(
            data=shap_explainer.shap_values(df),
            index=df.index,
            columns=df.columns)
        return preds, shaps, shap_expected_value

    def get_feature_contributions(y_true, y_pred, shap_values, shap_expected_value):
        """Compute prediction contribution and error contribution for each feature."""
        prediction_contribution = shap_values.abs().mean().rename("prediction_contribution")
        ind_log_loss = individual_log_loss(y_true=y_true, y_pred=y_pred).rename("log_loss")
        y_pred_wo_feature = shap_values.apply(
            lambda feature: shap_expected_value + shap_values.sum(axis=1) - feature
            ).map(shap_sum2proba)
        ind_log_loss_wo_feature = y_pred_wo_feature.apply(
            lambda feature: individual_log_loss(y_true=y_true, y_pred=feature))
        ind_log_loss_diff = ind_log_loss_wo_feature.apply(lambda feature: ind_log_loss - feature)
        error_contribution = ind_log_loss_diff.mean().rename("error_contribution").T
        return prediction_contribution, error_contribution
    
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)

    preds, shaps, shap_expected_value = get_preds_shaps(model=base_model, df=X)

    assert ((preds - (shap_expected_value + shaps.sum(axis=1)).apply(
        shap_sum2proba)).abs() < 1e-10).all()
    prediction_contribution, error_contribution = get_feature_contributions(
        y_true=y,
        y_pred=preds,
        shap_values=shaps,
        shap_expected_value=shap_expected_value
    )

    contributions = pd.concat([prediction_contribution, error_contribution], axis=1)
    most_important_features = contributions[(contributions["error_contribution"] < 0)].sort_values(
        "prediction_contribution", ascending=False).head(n_most_important_features)
    if prediction_contribution_relative_threshold is not None:
        most_important_features = most_important_features[
            most_important_features['prediction_contribution'] > 
            prediction_contribution_relative_threshold 
            * most_important_features['prediction_contribution'].max()
        ]
    return most_important_features
