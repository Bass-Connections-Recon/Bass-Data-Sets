import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LassoLars
from sklearn.linear_model import Lasso
import matplotlib.dates as mdates
import random
import itertools
from sklearn.linear_model import Lars
from statsmodels.discrete.discrete_model import NegativeBinomial
from statsmodels.genmod.families import Poisson
from statsmodels.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

#Load Data
acled_daily = pd.read_excel('acled_daily_20250806_v2.xlsx')
cumulative = pd.read_excel('cumulative_injuries.xlsx',
    usecols=['Total # of Injuries in Gaza (cumulative from 10/7/23)', 'Date']
)
cumulative['Interpolated Injuries'] = cumulative['Total # of Injuries in Gaza (cumulative from 10/7/23)'].interpolate(method='linear')
cumulative.rename(columns={'Date':'event_date', 'Total # of Injuries in Gaza (cumulative from 10/7/23)': 'Total_injuries', 'Interpolated Injuries':"Interpolated_injuries"}, inplace=True)

mergedData = pd.merge(cumulative, acled_daily, on='event_date', how='left')

# Create new row as a DataFrame
new_row = pd.DataFrame({
    'event_date': [pd.to_datetime('2023-10-06')],
    'Interpolated_injuries': [0],
    **{col: [0] for col in mergedData.columns if col not in ['event_date', 'Interpolated_injuries']}
})

# Append and sort
mergedData = pd.concat([new_row, mergedData], ignore_index=True)
mergedData = mergedData.sort_values('event_date').reset_index(drop=True)
mergedData["diff_interpolated_injuries"] = mergedData["Interpolated_injuries"].diff()
mergedData["diff_interpolated_injuries"] = mergedData["injuries"].combine_first(mergedData["diff_interpolated_injuries"])

#can uncomment in order to display the data table 
#print(mergedData)

mergedData['moving_front'] = (mergedData['avg_pct_new_zone_attacks'] > 0.351).astype(int)

#Ceasefire dates
ceasefire_start="2025-01-19"
ceasefire_end="2025-03-18"


# amalgamate at week level (if needed for visualization, data summary, etc.)
mergedData['week'] = mergedData['event_date'] - pd.to_timedelta(mergedData['event_date'].dt.weekday, unit='d')
avg_cols=['moving_front', 'early_conflict']
agg_dict = {
    col: 'mean' if col in avg_cols else 'sum'
    for col in mergedData.select_dtypes(include='number').columns
    if col != 'week'
}
mergedData_week = mergedData.groupby('week').agg(agg_dict).reset_index()
mergedData_week = mergedData_week.rename(columns={'week': 'event_date'})
mergedData.drop(columns='week', inplace=True)

#Model Training 
from scipy.optimize import minimize
from scipy.special import gammaln
import scipy.stats as stats

def fit_negative_binomial_model_positive(X_train, y_train, X_test, alpha_val=1.0):
    """
    Fit a Negative Binomial model with L2 regularization and non-negative coefficients.

    Parameters:
        X_train (DataFrame): Training predictors with constant column included.
        y_train (Series): Training target.
        X_test (DataFrame): Test predictors with constant column included.
        alpha_val (float): Regularization strength.

    Returns:
        model: Fitted model object with .params
        y_pred: Predictions on X_test.
    """
    X = X_train.values
    y = y_train.values
    n_features = X.shape[1]

    def neg_log_likelihood(params):
        eta = X @ params
        mu = np.exp(eta)
        alpha = alpha_val  # fixed dispersion
        ll = gammaln(y + 1 / alpha) - gammaln(1 / alpha) - gammaln(y + 1)
        ll += y * np.log(mu / (mu + 1 / alpha)) + (1 / alpha) * np.log(1 / (1 + alpha * mu))
        nll = -np.sum(ll)
        l2_penalty = alpha_val * np.sum(params**2)
        return nll + l2_penalty

    init_params = np.zeros(n_features)
    bounds = [(0, None)] * n_features  # force all coefficients to be ≥ 0

    result = minimize(
        neg_log_likelihood,
        init_params,
        method='L-BFGS-B',  # BFGS doesn't support bounds; use L-BFGS-B
        bounds=bounds
    )

    class PenalizedModel:
        def __init__(self, params):
            self.params = params

        def predict(self, X_input):
            return np.exp(X_input @ self.params)

    model = PenalizedModel(result.x)
    y_pred = model.predict(X_test.values)
    return model, y_pred



# summary table for the penalized model
import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm

def summarize_penalized_model(model, X, y, model_name="Penalized NB"):
    """
    Summarize a manually penalized model similar to statsmodels .summary().

    Parameters:
        model: Object with .params and .predict(X) methods
        X (pd.DataFrame): Training data used to fit model
        y (pd.Series or np.array): Observed target
        model_name (str): Name to display

    Returns:
        summary_df (pd.DataFrame): Coefficient table
        fit_stats (dict): Model statistics
    """
    coef = model.params
    mu = model.predict(X)  # predicted values
    residuals = y - mu
    n = len(y)
    k = X.shape[1]

    # Convert to NumPy for matrix ops
    X_np = X.to_numpy()
    W = np.diag(mu)
    
    # Estimate standard errors
    try:
        cov_matrix = np.linalg.pinv(X_np.T @ W @ X_np)
        std_err = np.sqrt(np.diag(cov_matrix))
    except np.linalg.LinAlgError:
        std_err = np.full_like(coef, np.nan)

    # Z-scores and p-values
    z_scores = coef / std_err
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))  # Correct two-tailed test


    # Confidence intervals
    ci_lower = coef - 1.96 * std_err
    ci_upper = coef + 1.96 * std_err

    # Assemble summary table
    summary_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": coef,
        "Std.Err": std_err,
        "z": z_scores,
        "P>|z|": p_values,
        "[0.025": ci_lower,
        "0.975]": ci_upper
    })

    # Log-likelihood (Poisson approx)
    ll = np.sum(y * np.log(mu + 1e-8) - mu - gammaln(y + 1))
    ll_null = np.sum(y * np.log(np.mean(y) + 1e-8) - np.mean(y) - gammaln(y + 1))
    pseudo_r2 = 1 - ll / ll_null
    aic = -2 * ll + 2 * k

    fit_stats = {
        "Model": model_name,
        "Log-Likelihood": ll,
        "Null Log-Likelihood": ll_null,
        "Pseudo R-squared": pseudo_r2,
        "AIC": aic,
        "n": n,
        "k": k
    }

    return summary_df, fit_stats

from sklearn.linear_model import Lars
from sklearn.preprocessing import StandardScaler
from statsmodels.api import GLM, add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from itertools import combinations
import re

##ALPHA estimator
def estimate_alpha_moments(y):
    """
    Estimate dispersion parameter alpha via method of moments.

    Parameters:
        y (pd.Series or np.array): Target variable (e.g., counts)

    Returns:
        float: Estimated alpha
    """
    y = np.asarray(y)
    sample_mean = np.mean(y)
    sample_var = np.var(y, ddof=1)  # sample variance

    if sample_var <= sample_mean:
        print("Warning: Data not overdispersed. Returning alpha = 0.")
        return 0.0

    alpha_hat = (sample_var - sample_mean) / (sample_mean ** 2)
    return alpha_hat


##force include pop2 estimates
def force_include_middle_pop(selected_features, all_features):
    """
    Ensure that if pop_1 and pop_3 of a category are included, pop_2 is too.
    
    Parameters:
        selected_features (list): Currently selected feature names.
        all_features (list): Full list of feature names (after cleaning).
    
    Returns:
        list: Updated selected_features list.
    """
    selected_set = set(selected_features)
    to_add = []

    # Find all features that end with _pop_1 or _pop_3
    pop1_features = [f for f in selected_set if re.search(r"-pop_1$", f)]
    pop3_features = [f for f in selected_set if re.search(r"-pop_3$", f)]

    # Extract base names and look for matching pop_2
    for f1 in pop1_features:
        base = f1.replace("-pop_1", "")
        f3 = base + "-pop_3"
        f2 = base + "-pop_2"

        if f3 in selected_set and f2 in all_features and f2 not in selected_set:
            to_add.append(f2)

    return sorted(list(selected_set.union(to_add)))



# ====== Model Builder ======
def prepare_and_model_negative_binomial(
    df, date_col, target_col,
    start_date, validation_start_date, end_date,
    base_features,
    lag_min =1000000, #due to sparsity of data, lagging of independent variables in model creation was not considered in final model (i.e. lag_min set to value above that present in dataset)
    frequency_cutoff=0,
    do_add_constant=False,
    vif_threshold=10,
    lars_features=10,
    do_vif=True,
    do_lars=True,
    do_interactions=False,
    interaction_base_cols=[],
    do_l2reg=False,
    alpha_val=1
):
    
    
    # Identify high-frequency predictors
    high_freq_features = [col for col in base_features if (df[col] != 0).sum() >= lag_min]

    # Add lagged versions of only high-frequency predictors
    for lag in [1, 2]:
        for col in high_freq_features:
            df[f'lag{lag}_{col}'] = df[col].shift(lag).fillna(0)


    # Extend base_features to include lagged versions
    lagged_features = [f'lag{lag}_{col}' for lag in [1, 2] for col in high_freq_features]
    base_features += lagged_features

    print(base_features)
    
    train_df = df[(df[date_col] >= start_date) & (df[date_col] < validation_start_date)]
    test_df = df[(df[date_col] >= end_date)]

    # Filter out features with average < frequency cutoff in train_df
    low_avg_cols = train_df[base_features].mean() < frequency_cutoff
    drop_cols = [col for col in low_avg_cols[low_avg_cols].index if col != "moving_front"]
    train_df = train_df.drop(columns=drop_cols)
    test_df = test_df.drop(columns=drop_cols)
    base_features = [f for f in base_features if f not in drop_cols]

    #drop duplicates within training dataset
    dup_features=train_df.T.duplicated()
    features_to_drop = dup_features[dup_features].index.tolist()
    train_df = train_df.drop(columns=features_to_drop)
    test_df = test_df.drop(columns=features_to_drop)
    base_features = [f for f in base_features if f not in features_to_drop]
    
    print("here are the base features included")
    print(base_features)
   
    y_train = train_df[target_col]
    y_test = test_df[target_col]
    X_train = train_df[base_features]
    X_test = test_df[base_features]

    print("here is the training X data")
    print(X_train)

    #estimate dispersion with alpha estimator
    alpha_val = estimate_alpha_moments(y_train)
    print(f"Estimated alpha: {alpha_val:.4f}")

    #add in interaction terms for selected variables
    if do_interactions:
        interaction_terms = []
        all_other_cols = [col for col in X_train.columns if col not in interaction_base_cols]
        for col1 in interaction_base_cols:
            for col2 in all_other_cols:
                inter_col = f"{col1}*{col2}"
                if inter_col not in X_train.columns:  # Avoid accidental overwrite
                    X_train[inter_col] = X_train[col1] * X_train[col2]
                    X_test[inter_col] = X_test[col1] * X_test[col2]
                    interaction_terms.append(inter_col)

        print(f"Added {len(interaction_terms)} interaction terms")

        
    #scale all variables
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    scaler_train=X_train
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    #drop heavily correlated variables
    if do_vif:
        def calculate_vif(df):
            vif_data = pd.DataFrame()
            vif_data["feature"] = df.columns
            vif_data["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
            return vif_data

        while True:
            vif = calculate_vif(X_train)
            max_vif = vif["VIF"].max()
            print(max_vif)
            if max_vif > vif_threshold:
                drop_feature = vif.loc[vif["VIF"] == max_vif, "feature"].values[0]
                X_train = X_train.drop(columns=drop_feature)
                X_test = X_test.drop(columns=drop_feature)
                print("drop feature:")
                print(drop_feature)
            else:
                break

    #perform LASSO with constraints (non-neg coefficients, alpha_val calculated above)
    if do_lars:
        # Use Lasso with non-negative coefficients
        lasso = Lasso(alpha=alpha_val, positive=True, max_iter=10000)
        lasso.fit(X_train, y_train)
        selected_features = list(X_train.columns[lasso.coef_ != 0])
        X_train_selected = X_train[selected_features]
        X_test_selected = X_test[selected_features]
    else:
        selected_features = X_train.columns
        X_train_selected = X_train
        X_test_selected = X_test
    
    # Drop perfectly collinear columns using QR decomposition
    from numpy.linalg import matrix_rank
    rank = matrix_rank(X_train_selected.values)
    if rank < X_train_selected.shape[1]:
        print("Warning: X matrix is rank deficient. Consider removing collinear features.")

    # Drop constant variable columns
    X_train_selected = X_train_selected.loc[:, (X_train_selected != X_train_selected.iloc[0]).any()]
    X_test_selected = X_test_selected[X_train_selected.columns]

    # Add in "constant" for modeling 
    if do_add_constant:
        X_train_selected = add_constant(X_train_selected, has_constant='add')
        X_test_selected = add_constant(X_test_selected, has_constant='add')

    #ensure columns in same order
    X_test_selected = X_test_selected[X_train_selected.columns]

    #can uncomment to display data 
    #print("training data going into model")
    #print(X_train_selected)
    #print(X_test_selected)

    # default negative binomial model if not doing L2 regularization
    model = NegativeBinomial(y_train, X_train_selected).fit()
    y_pred = model.predict(X_test_selected)

    #L2 regularization with positive coefficients and calculated alpha_val
    if do_l2reg:
        model, y_pred = fit_negative_binomial_model_positive(X_train_selected, y_train, X_test_selected, alpha_val)
        summary_df, fit_stats = summarize_penalized_model(model, X_train_selected, y_train)
        print(summary_df.to_string(index=False, float_format="%.4f"))
        print("\nModel Fit Statistics:")
        for k, v in fit_stats.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

    # === VALIDATION SET CALIBRATION ===
    val_df = df[(df[date_col] >= validation_start_date) & (df[date_col] < end_date)].copy()
    y_val = val_df[target_col].astype(float)
    
    # 1) Start from the exact columns the scaler was fit on (pre-VIF/LARS)
    scaler_cols = list(scaler.feature_names_in_)
    
    # 1a) Ensure any interaction terms present in scaler_cols exist in val_df
    for col in scaler_cols:
        if '*' in col and col not in val_df.columns:
            a, b = col.split('*', 1)
            if a in val_df.columns and b in val_df.columns:
                val_df[col] = val_df[a] * val_df[b]
            else:
                val_df[col] = 0.0
    
    # 2) Build full validation frame in the same column order
    X_val_full = val_df.reindex(columns=scaler_cols, fill_value=0.0)
    
    # 3) Scale using the training scaler
    X_val_full_s = pd.DataFrame(
        scaler.transform(X_val_full),
        columns=scaler_cols,
        index=X_val_full.index
    )
    
    # 4) Reduce to the model's selected design matrix (post-VIF/LARS, and const if applicable)
    model_cols_wo_const = [c for c in X_train_selected.columns if c != 'const']
    X_val_selected = X_val_full_s.reindex(columns=model_cols_wo_const, fill_value=0.0)
    if 'const' in X_train_selected.columns:
        X_val_selected = add_constant(X_val_selected, has_constant='add')
    X_val_selected = X_val_selected[X_train_selected.columns]
    
    # 5) Predict on validation and compute additive calibration
    mu_val = model.predict(X_val_selected)
    additive_offset = np.mean(y_val - mu_val)  # mean difference
    
    print(f"Additive calibration offset from validation: {additive_offset:.3f}")
    
    # 6) Apply additive calibration to test predictions
    y_pred_test_cal = np.clip(y_pred + additive_offset, 0, None)
    y_validation = np.clip(mu_val + additive_offset, 0, None)
    
    # Ceasefire zeroing (if applicable)
    mask = (test_df[date_col] >= ceasefire_start) & (test_df[date_col] <= ceasefire_end)
    y_pred_test_cal[mask.to_numpy()] = 0
    
    # store both
    y_pred_raw = y_pred.copy()
    y_pred = y_pred_test_cal

    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'val_df': val_df,
        "model": model,
        "selected_features": selected_features,
        "X_train": X_train_selected,
        "y_train": y_train,
        "X_validation": X_val_selected,
        "y_validation_actual": y_val,
        "y_validation": y_validation,
        "y_validation_raw": mu_val,
        "additive_offset": additive_offset,
        "X_test": X_test_selected,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_raw":y_pred_raw,
        "scaler": scaler,
        "scaler_train":scaler_train
    }

exclude_cols = ["injuries", "event_date","Date","diff_interpolated_injuries", "Interpolated_injuries","Total_injuries","territorial_expansion","new_zone_attacks","avg_pct_new_zone_attacks","pct_new_zone_attacks_14d"]
base_features = [col for col in mergedData.columns if col not in exclude_cols]
print(base_features)

result = prepare_and_model_negative_binomial(
            df=mergedData,
            date_col='event_date',
            target_col='diff_interpolated_injuries',
            start_date='2023-10-07',
            validation_start_date='2024-04-01',
            end_date='2024-05-01',
            base_features=base_features,
            lag_min=24000,
            frequency_cutoff=0.14,
            do_add_constant=True,
            do_vif=True, 
            do_lars=True,
            do_interactions=True,
            interaction_base_cols=['moving_front'],
            do_l2reg=True,
            alpha_val=0.6517
        )