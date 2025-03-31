import numpy as np
import pandas as pd
from typing import Dict, Any

from nvflare.apis.fl_context import FLContext

def combat_remote_step1(fl_ctx: FLContext, site_results: Dict[str, Any], agg_cache_dict: Dict[str, Any]):
    site_ids = list(site_results.keys())
    
    site_covar_list = [
        '{}_{}'.format('site', label) for index, label in enumerate(sorted(site_ids))    
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

def combat_remote_step2(fl_ctx: FLContext, site_results: Dict[str, Any], agg_cache_dict: Dict[str, Any]):
    sites = sorted(list(site_results.keys()))
    beta_vector_0 = [ np.array(site_results[site]["XtransposeX_local"], dtype=int) for site in sites]
    
    beta_vector_1 = sum(beta_vector_0)
    
    all_lambdas = [site_results[site]["lambda_value"] for site in sites]
    if np.unique(all_lambdas).shape[0] != 1:
        raise Exception("Unequal lambdas at local sites")
    
    beta_vector_1 = beta_vector_1 + np.unique(all_lambdas) * np.eye(beta_vector_1.shape[0])   

    beta_vectors = np.matrix.transpose(
    sum([
        np.matmul(np.linalg.inv(beta_vector_1),
                    site_results[site]["Xtransposey_local"])
        for site in site_results.keys()
    ]))
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