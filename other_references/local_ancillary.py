import warnings
import os
import numpy as np
import pandas as pd
import scipy as sp
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from numba import jit, prange

warnings.simplefilter("ignore")

def saveBin(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb+") as fh:
        header = "%s" % str(arr.dtype)
        for index in arr.shape:
            header += " %d" % index
        header += "\n"
        fh.write(header.encode())
        fh.write(arr.data.tobytes())
        os.fsync(fh)


def loadBin(path):
    with open(path, "rb") as fh:
        header = fh.readline().decode().split()
        dtype = header.pop(0)
        arrayDimensions = []
        for dimension in header:
            arrayDimensions.append(int(dimension))
        arrayDimensions = tuple(arrayDimensions)
        return np.frombuffer(fh.read(), dtype=dtype).reshape(arrayDimensions)


def mean_and_len_y(y):
    """Caculate the length mean of each y vector"""
    meanY_vector = y.mean(axis=0).tolist()
    lenY_vector = y.count(axis=0).tolist()

    return meanY_vector, lenY_vector


@jit(nopython=True)
def gather_local_stats(X, y):
    """Calculate local statistics"""
    size_y = y.shape[1]

    params = np.zeros((X.shape[1], size_y))
    sse = np.zeros(size_y)
    tvalues = np.zeros((X.shape[1], size_y))
    rsquared = np.zeros(size_y)

    for voxel in prange(size_y):
        curr_y = y[:, voxel]
        beta_vector = np.linalg.inv(X.T @ X) @ (X.T @ curr_y)
        params[:, voxel] = beta_vector

        curr_y_estimate = np.dot(beta_vector, X.T)

        SSE_global = np.linalg.norm(curr_y - curr_y_estimate)**2
        SST_global = np.sum(np.square(curr_y - np.mean(curr_y)))

        sse[voxel] = SSE_global
        r_squared_global = 1 - (SSE_global / SST_global)
        rsquared[voxel] = r_squared_global

        dof_global = len(curr_y) - len(beta_vector)

        MSE = SSE_global / dof_global
        var_covar_beta_global = MSE * np.linalg.inv(X.T @ X)
        se_beta_global = np.sqrt(np.diag(var_covar_beta_global))
        ts_global = beta_vector / se_beta_global

        tvalues[:, voxel] = ts_global

    return (params, sse, tvalues, rsquared, dof_global)


def local_stats_to_dict_vbm(X, y):
    """Wrap local statistics into a dictionary to be sent to the remote"""
    X1 = sm.add_constant(X).values.astype('float64')
    y1 = y.values.astype('float64')

    params, sse, tvalues, rsquared, dof_global = gather_local_stats(X1, y1)

    pvalues = 2 * sp.stats.t.sf(np.abs(tvalues), dof_global)

    keys = [
        "Coefficient", "Sum Square of Errors", "t Stat", "R Squared"
    ]

    values1 = pd.DataFrame(list(
        zip(params.T.tolist(), sse.tolist(), tvalues.T.tolist(),
            pvalues.T.tolist(), rsquared.tolist())),
                           columns=keys)

    local_stats_list = values1.to_dict(orient='records')

    beta_vector = params.T.tolist()

    return beta_vector, local_stats_list


def ignore_nans(X, y):
    """Removing rows containing NaN's in X and y"""

    if type(X) is pd.DataFrame:
        X_ = X.values.astype('float64')
    else:
        X_ = X

    if type(y) is pd.Series:
        y_ = y.values.astype('float64')
    else:
        y_ = y

    finite_x_idx = np.isfinite(X_).all(axis=1)
    finite_y_idx = np.isfinite(y_)

    finite_idx = finite_y_idx & finite_x_idx

    y_ = y_[finite_idx]
    X_ = X_[finite_idx, :]

    return X_, y_


def add_site_covariates(args, X, sample_count):
    """Add site specific columns to the covariate matrix"""
    # biased_X =pd.DataFrame(X)
    biased_X = X
    site_covar_list = args["input"]["site_covar_list"]

    site_matrix = np.zeros(( sample_count, len(site_covar_list)), dtype=int)
    site_df = pd.DataFrame(site_matrix, columns=site_covar_list)

    select_cols = [
        col for col in site_df.columns
        if args["state"]["clientId"] in col[len('site_'):]
    ]

    site_df[select_cols] = 1

    biased_X.reset_index(drop=True, inplace=True)
    site_df.reset_index(drop=True, inplace=True)

    augmented_X = pd.concat([biased_X, site_df], axis=1)
    return augmented_X

def interpolate_missing_data(Y,X):
    # Interpolate missing data within each site.
    # X = covariates
    # Y = data
    
    if np.isnan(Y).any(): # Y = data from batch

        for j in range(Y.shape[0]): # iterate over each feature in Y
            Y_j = Y[j, :] # Get data from the feature

            # if type(Y_j[0] ) == bool:
            #     Y_j = Y_j.astype(float)

            is_na = np.where(np.isnan(Y_j))[0] # Get indices of samples with nan values
            
            # If there is at least 1 nan value in the feature but
            # not all samples have a nan value for the feature and if there are covariates.
            # then interpolate using covariates
            if len(is_na) > 0 and len(is_na) < np.shape(Y)[1] and np.any(X):
                if len(is_na) == 1:
                    X_is_na = X[is_na, :].reshape(1, -1)
                else:
                    X_is_na = X[is_na, :]
                
                lm_model =  OLS(Y_j.astype(float).T, sm.tools.tools.add_constant(X.astype(float)),missing="drop").fit()
                beta = lm_model.params
                beta[np.isnan(beta)] = 0
                #print(beta)

                Y[j, is_na] = np.dot(np.hstack([np.ones((len(is_na), 1)), X_is_na]), beta)
            else: # otherwise, just use the mean of the feature
                Y[j, is_na] = np.nanmean(Y_j)
        
    return(Y)    

def identify_categorical_covariates(covariates):
    
    covariate_categories = [type(covariate_vals) for covariate_vals in list(covariates.values[0,:])]
    
    return covariate_categories

from sklearn.preprocessing import OneHotEncoder

def encode_covariates(covariates,covariate_categories):
    column_names = [];
    covariate_names = np.expand_dims(covariates.columns.values,axis=1)
    
    for idx, covariate_cat in enumerate(covariate_categories):
        if covariate_cat == str:
            one_hot_encoder = OneHotEncoder().fit(np.expand_dims(covariates.values[:,idx],axis=1))
            one_hot_results = one_hot_encoder.transform(np.expand_dims(covariates.values[:,idx],axis=1)).toarray()
            try: # accommodate different sklearn versions
                column_names.extend(one_hot_encoder.get_feature_names(covariate_names[idx]))
            except:
                column_names.extend(one_hot_encoder.get_feature_names_out(covariate_names[idx]))

            if idx == 0:
                new_covariates = list(one_hot_results)
            else:
                new_covariates = np.hstack((new_covariates,list(one_hot_results)))
        else:
            if idx == 0:
                new_covariates = np.array(list(np.expand_dims(covariates.values[:,idx],axis=1)),dtype=float)
            else:
                new_covariates = np.hstack((new_covariates,np.array(list(np.expand_dims(covariates.values[:,idx],axis=1)),dtype=float)))
            column_names.append(covariate_names[idx][0])
        
        new_covariates = pd.DataFrame(data=new_covariates,columns=column_names)
    return(new_covariates)