import copy
import math
import os
from typing import Dict
import pandas as pd
import numpy as np
import numpy.linalg as la

from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from nvflare.apis.fl_constant import FLContextKey

from utils.utils import get_data_directory_path, get_output_directory_path, get_computation_parameters
from . import local_ancillary as lc
from . import ancillary as ac

def parse_clientId(inp_str):
    num = ""
    for c in inp_str:
        if c.isdigit():
            num = num + c
    return int(num)

def convert_zeroes(x):
    x[x==0] = 1
    return x

def aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat,ddof=1)
    return (2 * s2 +m**2) / float(s2)

def bprior(delta_hat):
    m = delta_hat.mean()
    s2 = np.var(delta_hat,ddof=1)
    return (m*s2+m**3)/s2

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def convert_zeroes(x):
    x[x==0] = 1
    return x

def int_eprior(sdat, g_hat, d_hat):
    r = sdat.shape[0]
    gamma_star, delta_star = [], []
    for i in range(0,r,1):
        g = np.delete(g_hat,i)
        d = np.delete(d_hat,i)
        x = sdat[i,:]
        n = x.shape[0]
        j = np.repeat(1,n)
        A = np.repeat(x, g.shape[0])
        A = A.reshape(n,g.shape[0])
        A = np.transpose(A)
        B = np.repeat(g, n)
        B = B.reshape(g.shape[0],n)
        resid2 = np.square(A-B)
        sum2 = resid2.dot(j)
        LH = 1/(2*math.pi*d)**(n/2)*np.exp(-sum2/(2*d))
        LH = np.nan_to_num(LH)
        gamma_star.append(sum(g*LH)/sum(LH))
        delta_star.append(sum(d*LH)/sum(LH))
    adjust = (gamma_star, delta_star)
    return adjust


def find_non_parametric_adjustments(s_data, LS):
    gamma_star, delta_star = [], []
    temp = int_eprior(s_data, LS['gamma_hat'], LS['delta_hat'])
    gamma_star.append(temp[0])
    delta_star.append(temp[1])
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)
    # raise Exception(gamma_star.shape, delta_star.shape)
    return gamma_star, delta_star


def adjust_data_final(s_data, batch_design, gamma_star, delta_star, stand_mean, mod_mean, var_pooled, dat, local_n_sample):
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)
    dsq = np.sqrt(delta_star)
    denom = np.dot(dsq.T, np.ones((1, local_n_sample)))
    numer = np.array(bayesdata  - np.dot(batch_design, gamma_star).T)
    bayesdata = numer / denom
    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))

    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, local_n_sample))) + stand_mean + mod_mean
    return bayesdata

def list_recursive(d, key):
    for k, v in d.items():
        if isinstance(v, dict):
            for found in list_recursive(v, key):
                yield found
        if k == key:
            yield v

def csv_parser(file_url):
    dataFrame = pd.read_csv(file_url)
    data_url = dataFrame["data_url"][0]
    covar_url = dataFrame["covar_info"][0]
    return  data_url, covar_url

def folders_in(path_to_parent):
    for fname in os.listdir(path_to_parent):
        if os.path.isdir(os.path.join(path_to_parent,fname)):
            yield os.path.join(path_to_parent,fname)


def perform_task_step1(fl_ctx: FLContext):
    input_directory = get_data_directory_path(fl_ctx)
    
    computation_parameters = get_computation_parameters(fl_ctx)
    covariate_file_name = computation_parameters['covariate_file']
    data_file_name = computation_parameters['data_file']
    
    covariates_path = os.path.join(input_directory, covariate_file_name)
    data_path = os.path.join(input_directory,data_file_name)
    combat_type = computation_parameters['combat_alg_type']
        
    # log(fl_ctx, f'-- Checking file paths : {str(covariates_path)} and {str(data_path)}')
    cache_dict = {
        'covar_urls': covariates_path,
        'data_urls': data_path,
        'lambda_value': 0,
        'combat_alg_type': combat_type
    }
    
    # log(fl_ctx, f'step1: {cache_dict}')
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
    
    # log(fl_ctx, f'step2: initial: {covar_url}, {data_url}, {lambda_value}, {combat_alg_type}')
    
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
    
    # log(fl_ctx, f'site_name: {site_name} and site_index: {site_index}')
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


def perform_task_step3(sharebale: Shareable, fl_ctx: FLContext, abort_signal: Signal, cache_dict: Dict):
    #TODO: check the cache directory
    output_directory = get_output_directory_path(fl_ctx)
    agg_results = sharebale.get('result')
    
    covar = ac.loadBin(os.path.join(output_directory, "X.npy"))
    mat_Y = ac.loadBin(os.path.join(output_directory, "Y.npy"))
    
    data = np.array(np.transpose(mat_Y))
    data_columns = pd.read_json(cache_dict.get("data_names"), orient='split').values
    
    # log(fl_ctx, f'data_columns: {data_columns}')
    
    design = covar
    B_hat = np.array(agg_results["B_hat"])
    n_sample = agg_results["n_sample"]
    n_batch = agg_results["n_batch"]
    site_array = agg_results["site_array"]
    local_n_sample = cache_dict.get("local_sample_count")
    
    local_var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((local_n_sample, 1)) / float(n_sample))
    stand_mean = np.array(agg_results["stand_mean"])
    mod_mean = 0
    
    if design is not None:
        tmp = copy.deepcopy(design)
        tmp[:,range(-n_batch,0)] = 0
        mod_mean = np.transpose(np.dot(tmp, B_hat))
    
    cache_dict = {
        "mod_mean": mod_mean.tolist(),
        "stand_mean": stand_mean.tolist(),
        "site_array": site_array,
        "n_batch": n_batch,
        # "covar_urls": covar_url
    }
    
    output_dict = {
       "local_var_pooled": local_var_pooled.tolist(),
       "computation_phase": "local_2",
       "data_names":pd.DataFrame(data_columns).to_json(orient='split')
       # "data": pd.DataFrame(data).to_json(orient='split'),
    }
    
    result = {
        'output': output_dict,
        'cache': cache_dict
    }
    
    return result
    

def perform_task_step4(sharebale: Shareable, fl_ctx: FLContext, abort_signal: Signal, cache_dict: Dict):
    #TODO: check the cache directory
    output_directory = get_output_directory_path(fl_ctx)
    agg_results = sharebale.get('result')
    
    var_pooled = np.array(agg_results['global_var_pooled'])
    data = np.array(np.transpose(ac.loadBin(os.path.join(output_directory, 'Y.npy'))))
    stand_mean = np.array(cache_dict.get('stand_mean')).T
    mod_mean = np.array(cache_dict.get('mod_mean'))
    local_n_sample = cache_dict.get('local_sample_count')
    
    site_index = cache_dict.get('site_index')
    site_array = cache_dict.get('site_array')
    
    # log(fl_ctx, f'site4: stand_mean: {stand_mean}, mod_mean: {mod_mean}, local_n_samples: {local_n_sample}, site_index: {site_index}, site_array: {site_array}')
    
    indices = [index for index,element in enumerate(site_array) if element== int(site_index)]
    filtered_mean = stand_mean[indices]
    local_stand_mean = filtered_mean.T
    s_data = ((data - local_stand_mean - mod_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, local_n_sample))))
    
    batch_design = np.array([[1]*local_n_sample]).T
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)
    delta_hat = []
    delta_hat.append(np.var(s_data, axis=1, ddof=1))
    delta_hat = list(map(convert_zeroes, delta_hat))
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat, axis=1, ddof=1)
    
    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))
    LS_dict = {}
    LS_dict['gamma_hat'] = gamma_hat
    LS_dict['delta_hat'] = delta_hat
    LS_dict['gamma_bar'] = gamma_bar
    LS_dict['t2'] = t2
    LS_dict['a_prior'] = a_prior
    LS_dict['b_prior'] = b_prior
    gamma_star, delta_star = find_non_parametric_adjustments(s_data, LS_dict)
    bayesdata = adjust_data_final(s_data, batch_design,gamma_star, delta_star, local_stand_mean, mod_mean, var_pooled, data, local_n_sample)
    harmonized_data = np.transpose(bayesdata)
    
    mat_Y_labels = pd.read_json(cache_dict.get('data_names'), orient='split').values
    
    df = pd.DataFrame(harmonized_data, columns=mat_Y_labels)
    output_name = 'harmonized_site_'+str(site_index)+'_data.csv'
    
    df.to_csv(os.path.join(output_directory, output_name), index=False)
