import math
import os
import json
import pandas as pd
import numpy as np
import copy
import numpy.linalg as la
from typing import List, Dict

from local_ancillary import identify_categorical_covariates, encode_covariates, interpolate_missing_data, saveBin, loadBin

# INPUTSPEC VALUES
BASE_INPUT_DATA = "test/input/{site}/simulatorRun"
BASE_OUTPUT_DIR = "test/output"
BASE_CACHE_DIR = "test/cache"
SUB_COVARIATES = "CatCovariate.csv"
SUB_DATA = "MissingData.csv"
COMBAT_TYPE = "combatMegaDC" # ["combatMegaDC", "combatDC"]

# VALUES which are not in inputspec but qualified for it.
LAMBDA = 0

AGGREGATOR_CACHE: Dict[str, any] = {} # REMOTE CACHE
EXECUTOR_CACHE: Dict[str, dict] = {}  # LOCAL CACHE

############ LOCAL HELPER FUNCTIONS ################
def parse_clientId(inp_str):
    num = ""
    for c in inp_str:
        if c.isdigit():
            num = num + c
    return int(num) + 1

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


def add_site_covariates(agg_results, site_name, X, sample_count):
    """Add site specific columns to the covariate matrix"""
    biased_X = X
    site_covar_list = agg_results['site_covar_list']
    site_matrix = np.zeros(( sample_count, len(site_covar_list)), dtype=int)
    site_df = pd.DataFrame(site_matrix, columns=site_covar_list)

    select_cols = [
        col for col in site_df.columns
        if site_name in col[len('site_'):]
    ]

    site_df[select_cols] = 1

    biased_X.reset_index(drop=True, inplace=True)
    site_df.reset_index(drop=True, inplace=True)

    augmented_X = pd.concat([biased_X, site_df], axis=1)
    return augmented_X


def perform_local_step1(site_name: str, covariates_path: str, data_path: str):
    EXECUTOR_CACHE.setdefault(site_name, {})
    EXECUTOR_CACHE[site_name].update({
        "data_urls": data_path, 
        "covar_urls": covariates_path, 
        "lambda_value": LAMBDA, 
        "combat_alg_type": COMBAT_TYPE
    })
    
def perform_remote_step1(sites: List[str]):    
    site_covar_list = [
        '{}_{}'.format('site', label) for _, label in enumerate(sorted(sites))    
    ]
    
    agg_results = {
        "site_covar_list": sorted(site_covar_list),
    }
    
    return agg_results
    

def perform_local_step2(site_name: str, agg_results: Dict[str, any]):
    covar_url = EXECUTOR_CACHE[site_name].get("covar_urls")
    data_url = EXECUTOR_CACHE[site_name].get("data_urls")
    lambda_value = EXECUTOR_CACHE[site_name].get("lambda_value")
    combat_alg_type = EXECUTOR_CACHE[site_name].get("combat_alg_type")

    # if a covariates URL was passed and the file has any stored information
    if len(covar_url) > 0 and os.path.getsize(covar_url): 
        mat_X = pd.read_csv(covar_url) # covariates

        # get covariate types
        X_cat = identify_categorical_covariates(mat_X)

        # if there are categorical covariates, one-hot-encode them
        if str in X_cat:
            mat_X = encode_covariates(mat_X,X_cat)
    else:
        mat_X = pd.DataFrame()
    
    mat_Y = pd.read_csv(data_url) # data
    site_index = parse_clientId(site_name)
    X = mat_X 
    Y = mat_Y.values
    Y_columns = mat_Y.columns 
    sample_count = len(Y)

    # Interpolation Missing Data
    if combat_alg_type == "combatMegaDC":
        Y = interpolate_missing_data(Y.T,X.to_numpy()).T # convert nan values to NULL/None .replace({np.nan: None}
    
    # Save Data File
    saveBin(os.path.join(BASE_CACHE_DIR, site_name, "Y.npy"), Y)

    # Add Site Data to Covariates to Prep for COMBAT
    augmented_X = add_site_covariates(agg_results, site_name, X, sample_count)
    biased_X = augmented_X.values 
    
    # Save Covariate File
    # saveBin(os.path.join(cache_dir, "X.npy"), np.array(list(X.to_numpy()),dtype=float))
    saveBin(os.path.join(BASE_CACHE_DIR, site_name, "X.npy"), np.array(list(augmented_X.to_numpy()),dtype=float))

    XtransposeX_local = np.matmul(np.matrix.transpose(biased_X), biased_X)
    Xtransposey_local = np.matmul(np.matrix.transpose(biased_X), Y)

    
    EXECUTOR_CACHE[site_name].update({
        # "covariates": augmented_X.to_json(orient='split'),
        "local_sample_count": sample_count,
        # "covar_urls": covar_url,
        # "data": pd.DataFrame(Y).to_json(orient='split'),
        "data_names":pd.DataFrame(Y_columns).to_json(orient='split'),
        "site_index": site_index,
        "combat_alg_type": combat_alg_type
    })
    
    return {
        "local_sample_count": sample_count,
        "XtransposeX_local": XtransposeX_local.tolist(),
        "Xtransposey_local": Xtransposey_local.tolist(),
        "lambda_value": lambda_value,
        "site_index": site_index,
    }


def perform_remote_step2(site_results: Dict[str, any]):
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

    AGGREGATOR_CACHE.update({
        "avg_beta_vector": B_hat.tolist(),
        "stand_mean": stand_mean.tolist(),
        "grand_mean": grand_mean.tolist()
    })
    
    return agg_results


def perform_local_step3(site_name: str, agg_results: Dict[str, any]):
    
    # covar_url =  args["cache"]["covar_urls"]
    # covar = pd.read_json(cache_list["covariates"], orient='split')
    covar = loadBin(os.path.join(BASE_CACHE_DIR, site_name, "X.npy"))

    # it imports data again
    mat_Y = loadBin(os.path.join(BASE_CACHE_DIR, site_name, "Y.npy")) #pd.read_json(cache_list["data"], orient='split')
    data = np.array(np.transpose(mat_Y))# np.array(np.transpose(mat_Y.values))
    data_columns = pd.read_json(EXECUTOR_CACHE[site_name].get("data_names"), orient='split').values

    design = covar#.values
    B_hat = np.array(agg_results["B_hat"])
    n_sample = agg_results["n_sample"]
    n_batch = agg_results["n_batch"]
    site_array = agg_results["site_array"]
    local_n_sample = EXECUTOR_CACHE[site_name]["local_sample_count"]
    # local_var_pooled_part1 = np.dot(design, B_hat).T
    local_var_pooled = np.dot(((data - np.dot(design, B_hat).T)**2), np.ones((local_n_sample, 1)) / float(n_sample))
    stand_mean = np.array(agg_results["stand_mean"])
    mod_mean = 0

    if design is not None:
        tmp = copy.deepcopy(design)
        tmp[:,range(-n_batch,0)] = 0
        mod_mean = np.transpose(np.dot(tmp, B_hat))

    EXECUTOR_CACHE[site_name].update({
        "mod_mean": mod_mean.tolist(),
        "stand_mean": stand_mean.tolist(),
        "site_array": site_array,
        "n_batch": n_batch,
        # "covar_urls": covar_url
    })
    
    return {
        "local_var_pooled": local_var_pooled.tolist(),
        #"data": pd.DataFrame(data).to_json(orient='split'),
        "data_names":pd.DataFrame(data_columns).to_json(orient='split')
    }
    
def perform_remote_step3(site_results: Dict[str, any]):
    sites=list(site_results.keys())
    var_pooled = [ np.array(site_results[site]["local_var_pooled"]) for site in sorted(sites)]
    global_var_pooled= sum(var_pooled)
    agg_results = {
        "global_var_pooled": global_var_pooled.tolist(),
    }
    return agg_results

def perform_local_step4(site_name: str, agg_results: Dict[str, any]):
    var_pooled = np.array(agg_results["global_var_pooled"])
    data =  np.array(np.transpose(loadBin(os.path.join(BASE_CACHE_DIR, site_name, "Y.npy"))))#pd.read_json(cache_list["data"], orient='split').values.T
    stand_mean = np.array(EXECUTOR_CACHE[site_name]["stand_mean"]).T
    mod_mean = np.array(EXECUTOR_CACHE[site_name]["mod_mean"])
    local_n_sample = EXECUTOR_CACHE[site_name]["local_sample_count"]

    site_index = EXECUTOR_CACHE[site_name]["site_index"]
    site_array = EXECUTOR_CACHE[site_name]["site_array"]
    indices = [name for name, element in enumerate(site_array) if element == int(site_index)]
    filtered_mean = stand_mean[indices]
    local_stand_mean = filtered_mean.T
    s_data = ((data - local_stand_mean - mod_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, local_n_sample))))
    # covar = loadBin(os.path.join(cache_dir, "X.npy")) #pd.read_json(cache_list["covariates"], orient='split')
    # design = covar#.values
    # n_batch = cache_list["n_batch"]
    batch_design = np.array([[1]*local_n_sample]).T
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)
    delta_hat = []
    delta_hat.append(np.var(s_data,axis=1,ddof=1))
    delta_hat = list(map(convert_zeroes,delta_hat))
    gamma_bar = np.mean(gamma_hat, axis=1)
    t2 = np.var(gamma_hat,axis=1, ddof=1)

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
    bayesdata = adjust_data_final(s_data, batch_design, gamma_star, delta_star, local_stand_mean, mod_mean, var_pooled,  data, local_n_sample)
    harmonized_data = np.transpose(bayesdata)
    
    # covar_url =  args["cache"]["covar_urls"]
    mat_Y_labels = pd.read_json(EXECUTOR_CACHE[site_name]["data_names"], orient='split').values
    
    df = pd.DataFrame(harmonized_data, columns=mat_Y_labels)
    # df = pd.DataFrame(np.transpose(data), columns=mat_Y_labels) # can use this to check interpolation

    df.to_csv(os.path.join(BASE_OUTPUT_DIR, 'harmonized_site_'+str(site_name)+'_data.csv'),  index=False)

    return {
       "message": "Data Harmonization complete",
    }


### TODO: check whether this is needed
def perform_remote_step4():
    return {"status": "Complete"}


def run_federated_combat():
    sites = ['local0', 'local1'] # check for consistency

	#Iteration - 1
    for site in sites:
        #covariates_path = os.path.join(f'../test_data/{site}/covariates.csv')
        #data_path = os.path.join(f'../test_data/{site}/data.csv')
        
        base_path = BASE_INPUT_DATA.format(site=site)
        covariates_path = os.path.join(base_path, SUB_COVARIATES)
        data_path = os.path.join(base_path, SUB_DATA)
        perform_local_step1(site, covariates_path, data_path)
    agg_results = perform_remote_step1(sites)
    
    site_results = {}
    for site in sites:
        site_results[site] = perform_local_step2(site, agg_results)
    agg_results = perform_remote_step2(site_results)

    for site in sites:
        site_results[site] = perform_local_step3(site, agg_results)
    agg_results = perform_remote_step3(site_results)
    
    for site in sites:
        site_results[site] = perform_local_step4(site, agg_results)
    perform_remote_step4()

run_federated_combat()



