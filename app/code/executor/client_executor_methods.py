import os
from typing import Dict
import pandas as pd
import numpy as np

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.fl_constant import FLContextKey

from utils.utils import get_data_directory_path, get_output_directory_path, log, get_computation_parameters
from . import local_ancillary as lc
from . import ancillary as ac

def parse_clientId(inp_str):
    num = ""
    for c in inp_str:
        if c.isdigit():
            num = num + c
    return int(num)

def perform_task_step1(fl_ctx: FLContext):
    input_directory = get_data_directory_path(fl_ctx)
    
    computation_parameters = get_computation_parameters(fl_ctx)
    covariate_file_name = computation_parameters['covariate_file']
    data_file_name = computation_parameters['data_file']
    
    covariates_path = os.path.join(input_directory, covariate_file_name)
    data_path = os.path.join(input_directory,data_file_name)
    combat_type = computation_parameters['combat_alg_type']
        
    log(fl_ctx, f'-- Checking file paths : {str(covariates_path)} and {str(data_path)}')
    cache_dict = {
        'covar_urls': covariates_path,
        'data_urls': data_path,
        'lambda_value': 0,
        'combat_alg_type': combat_type
    }
    
    result = {
        'cache': cache_dict,
        'output': {}
    }
    
    return result
    

def perform_task_step2(sharebale: Shareable, fl_ctx: FLContext, abort_signal: Signal, cache_dict: Dict):
    agg_results = sharebale.get('result')
    covar_url = cache_dict.get('covar_urls')
    data_url = cache_dict.get('data_urls')
    lambda_value = cache_dict.get('lambda_value')
    combat_alg_type = cache_dict.get('combat_alg_type')
    
    # if a covariates URL was passed and the file has any stored information
    if len(covar_url) > 0 and os.path.getsize(covar_url): 
        mat_X = pd.read_csv(covar_url) # covariates

        # get covariate types
        X_cat = lc.identify_categorical_covariates(mat_X)

        # if there are categorical covariates, one-hot-encode them
        if str in X_cat:
            mat_X = lc.encode_covariates(mat_X,X_cat)
    else:
        mat_X = pd.DataFrame()
    
    mat_Y = pd.read_csv(data_url) # data
    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
    site_index = parse_clientId(site_name)
    X = mat_X 
    Y = mat_Y.values
    Y_columns = mat_Y.columns 
    sample_count = len(Y)
    
    #TODO: check the cache directory
    output_directory = get_output_directory_path(fl_ctx)

    # Interpolation Missing Data
    if combat_alg_type == "combatMegaDC":
        Y = lc.interpolate_missing_data(Y.T,X.to_numpy()).T # convert nan values to NULL/None .replace({np.nan: None}
    
    # Save Data File
    ac.saveBin(os.path.join(output_directory, "Y.npy"), Y)

    # Add Site Data to Covariates to Prep for COMBAT
    augmented_X = lc.add_site_covariates(agg_results, site_name, X, sample_count)
    biased_X = augmented_X.values 

    # Save Covariate File
    # saveBin(os.path.join(cache_dir, "X.npy"), np.array(list(X.to_numpy()),dtype=float))
    ac.saveBin(os.path.join(output_directory, "X.npy"), np.array(list(augmented_X.to_numpy()),dtype=float))

    XtransposeX_local = np.matmul(np.matrix.transpose(biased_X), biased_X)
    Xtransposey_local = np.matmul(np.matrix.transpose(biased_X), Y)

    output_dict = {
        "local_sample_count": sample_count,
        "XtransposeX_local": XtransposeX_local.tolist(),
        "Xtransposey_local": Xtransposey_local.tolist(),
        "lambda_value": lambda_value,
        "site_index": site_index,
    }
    cache_dict = {
        "local_sample_count": sample_count,
        "data_names":pd.DataFrame(Y_columns).to_json(orient='split'),
        "site_index": site_index,
        "combat_alg_type": combat_alg_type
    }

    result = {"output": output_dict, "cache": cache_dict}

    return result


def perform_task_step3():
    pass

def perform_task_step4():
    pass
