import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

def run_dfm(target_df_train, target_df_test, reference_date):
    # Differencing training data to make it stationary
    target_train_diff = target_df_train.diff().dropna()
    
    # Selecting the optimal number of common factors based on AIC criterion
    aic_values = {}
    max_factors = min(len(target_df_train.columns), 5)
    for k in range(1, max_factors + 1):
        try:
            mod = DynamicFactor(target_train_diff, k_factors=k, factor_order=1)
            res = mod.fit(disp=False)
            aic_values[k] = res.aic
            print(f'Number of common factors {k}: AIC = {res.aic:.2f}')
        except Exception as e:
            print(f'Model fit failed with common factors {k}: {e}')
    optimal_factors = min(aic_values, key=aic_values.get)
    print(f'\nOptimal number of common factors: {optimal_factors}')
    
    # Fitting the optimal DFM
    nfm = DynamicFactor(target_train_diff, k_factors=optimal_factors, factor_order=1)
    nfm_result = nfm.fit(method="powell", maxiter=1000)
    
    shared_component = nfm_result.factors.smoothed.T
    shared_component = np.vstack([shared_component[0], shared_component])
    loadings = nfm_result.params.filter(like="loading").values.reshape(target_df_train.shape[1], optimal_factors)
    
    shared_component_original = shared_component.cumsum(axis=0)
    shared_component_final = shared_component_original @ loadings.T
    
    # Scaling to match the distribution of the training data
    scale_factor = (target_df_train.std() / shared_component_final.std()).values.reshape(1, -1)
    shared_component_final = shared_component_final * scale_factor + target_df_train.mean().values # Error
    
    # Calculating offset based on the reference date
    reference_index = target_df_train.index.get_loc(reference_date)
    offset = target_df_train.iloc[reference_index].values - shared_component_final[reference_index]
    shared_component_final = shared_component_final + offset
    idiosyncratic_component_final = target_df_train - shared_component_final
    
    # Processing test phase
    target_test_diff = target_df_test.diff().dropna()
    nfm_test_result = nfm_result.apply(target_test_diff)
    shared_component_test = nfm_test_result.factors.smoothed.T
    shared_component_test = np.vstack([shared_component_original[-1, :], shared_component_test])
    shared_component_test_original = shared_component_test.cumsum(axis=0)
    shared_component_test_final = shared_component_test_original @ loadings.T
    shared_component_test_final = shared_component_test_final * scale_factor + target_df_test.mean().values + offset
    idiosyncratic_component_test_final = target_df_test - shared_component_test_final
    
    # Combining shared components of train and test datasets
    combined_array = np.vstack((shared_component_original, shared_component_test_original))
    
    results = {
        "optimal_factors": optimal_factors,
        "nfm_result": nfm_result,
        "loadings": loadings,
        "scale_factor": scale_factor,
        "offset": offset,
        "shared_component_original": shared_component_original,
        "shared_component_final_train": shared_component_final,
        "idiosyncratic_component_train": idiosyncratic_component_final,
        "shared_component_final_test": shared_component_test_final,
        "idiosyncratic_component_test": idiosyncratic_component_test_final,
        "combined_shared_component": combined_array
    }
    return results
