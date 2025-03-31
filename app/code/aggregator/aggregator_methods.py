from nvflare.apis.fl_context import FLContext
from typing import Dict, Any

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