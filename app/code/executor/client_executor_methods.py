import os

from typing import Dict
from nvflare.apis.fl_context import FLContext
from utils.utils import get_data_directory_path, get_output_directory_path, log, get_computation_parameters

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
    

def perform_task_step2():
    pass

def perform_task_step3():
    pass

def perform_task_step4():
    pass

