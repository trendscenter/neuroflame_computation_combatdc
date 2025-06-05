from typing import Dict, Any
import numpy as np

from utils.logger import NvFlareLogger

def combat_remote_step1(site_results: Dict[str, Any]):
    site_ids = list(site_results.keys())
    
    site_covar_list = [
        '{}_{}'.format('site', label) for _, label in enumerate(sorted(site_ids))    
    ]
    
    output_dict = {
        'site_covar_list': sorted(site_covar_list)
    }
    
    cache_dict = {}
    
    results = {
        'output': output_dict,
        'cache': cache_dict
    }
    
    return results

def combat_remote_step2(site_results: Dict[str, Any], agg_cache_dict: Dict[str, Any], logger: NvFlareLogger):
    sites = sorted(list(site_results.keys()))
    beta_vector_0 = [ np.array(site_results[site]["XtransposeX_local"]) for site in sites]
    logger.debug('beta_vector_0: ', beta_vector_0)
    beta_vector_1 = sum(beta_vector_0)
    
    all_lambdas = [site_results[site]["lambda_value"] for site in sites]
    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")
    
    beta_vector_1 = beta_vector_1 + np.unique(all_lambdas) * np.eye(beta_vector_1.shape[0])   
    logger.debug('beta_vector_1: ', beta_vector_1)
    
    assert isinstance(beta_vector_1, np.ndarray), "beta_vector_1 must be a NumPy array"
    assert beta_vector_1.shape == (6, 6), f"Expected beta_vector_1.shape == (6,6), but got {beta_vector_1.shape}"

    
    # 2) Compute the explicit inverse of β₁ once.
    inv_beta = np.linalg.inv(beta_vector_1)
    logger.debug('inv_beta: ', inv_beta)
    #    inv_beta.shape == (6, 6)

    # 3) Initialize a (6×17) accumulator for summing each site’s contribution.
    sum_matrix = np.zeros((6, 17), dtype=float)

    # 4) Loop over sites in a deterministic order; for each site, do exactly:
    #      inv_beta @ XTy_local  (shape 6×6 @ shape 6×17 → shape 6×17),
    #    then add that to sum_matrix.
    for site in sorted(site_results.keys()):
        logger.debug('site: ', site)
        
        XTy_local = np.asarray(site_results[site]["Xtransposey_local"])
        
        logger.debug('  XTy_local: ', XTy_local)
        assert isinstance(XTy_local, np.ndarray), (
            f"site_results[{site}]['Xtransposey_local'] must be a NumPy array"
        )
        assert XTy_local.shape == (6, 17), (
            f"For site '{site}', expected XTy_local.shape == (6,17), but got {XTy_local.shape}"
        )
        # 4a) Compute inv_beta @ XTy_local exactly as in the original code.
        solved = inv_beta @ XTy_local
        #    solved.shape == (6, 17)
        logger.debug('  solved: ', solved)
        
        # 4b) Accumulate into sum_matrix (still shape 6×17).
        sum_matrix += solved
        logger.debug('  sum_matrix: ', sum_matrix)

    # 5) After summing all sites, sum_matrix is the elementwise sum of each inv(β₁) @ XTy_local.
    #    Finally, transpose to get shape (17×6) just like your original:
    #       beta_vectors = np.matrix.transpose(sum([...]))
    beta_vectors = sum_matrix.T  # shape = (17, 6)

    # 6) logger.debug or return beta_vectors.
    logger.debug(beta_vectors)
    B_hat = beta_vectors.T

    n_batch =  len(sites)
    
    sample_per_batch = np.array([ site_results[site]["local_sample_count"] for site in sites])

    n_sample = sum(site_results[site]["local_sample_count"] for site in sites)
    
    site_array = []
    for site in sites:
        site_array = np.concatenate((site_array, [int(site_results[site]["site_index"])]*int(site_results[site]["local_sample_count"])), axis=0)
    
    grand_mean = np.dot((sample_per_batch/ float(n_sample)).T, B_hat[-n_batch:,:])
    # raise Exception(grand_mean, grand_mean.shape)
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, n_sample)))
    
    agg_results = {
        "n_batch": n_batch,
        "B_hat": B_hat.tolist(),
        "n_sample": n_sample, 
        "grand_mean": grand_mean.tolist(),
        "stand_mean": stand_mean.tolist(),
        "site_array": site_array.tolist(),
    }
    logger.debug('agg_results: ', agg_results)

    agg_cache_dict.update({
        "avg_beta_vector": B_hat.tolist(),
        "stand_mean": stand_mean.tolist(),
        "grand_mean": grand_mean.tolist()
    })
    
    results = {
        'output': agg_results,
        'cache': agg_cache_dict
    }
    
    return results

def combat_remote_step3(site_results: Dict[str, Any], agg_cache_dict: Dict[str, Any], logger: NvFlareLogger):
    site_keys = list(site_results.keys())
    sorted_site_keys = sorted(site_keys)
    
    var_pooled = [ np.array(site_results[site]["local_var_pooled"]) for site in sorted_site_keys]
    logger.debug('var_pooled: ', var_pooled)
    global_var_pooled = sum(var_pooled)
    logger.debug('global_var_pooled: ', global_var_pooled)
    agg_results = {
        "global_var_pooled": global_var_pooled.tolist(),
    }
    
    results = {
        'output': agg_results,
        'cache': agg_cache_dict
    }
    
    return results