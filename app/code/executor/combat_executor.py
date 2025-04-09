import logging
import json
import os

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal

from utils.logger import NvFlareLogger
from utils.utils import get_computation_parameters, get_data_directory_path, get_output_directory_path
from utils.task_constants import *

from . import client_cache_store as cache
from . import client_executor_methods as helpers
from utils.types import ConfigDTO

class DCCombatExecutor(Executor):

    def __init__(self):
        """
        Initialize the Combat. This constructor sets up the logger.
        """
        super().__init__()
        logging.info('DC-Combat is running')

    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        """
        Main execution entry point. Routes tasks to specific methods based on the task name.
        
        Parameters:
            task_name: Name of the task to perform.
            shareable: Shareable object containing data for the task.
            fl_ctx: Federated learning context.
            abort_signal: Signal object to handle task abortion.
            
        Returns:
            A Shareable object containing results of the task.
        """
        
        client_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
        client_log_name = client_name+".log"
        output_path = get_output_directory_path(fl_ctx)
        
        # TODO: get logger log level from env
        logger = NvFlareLogger(client_log_name, output_path, 'info')
        
        cache_dict = cache.CacheSerialStore(get_output_directory_path(fl_ctx)) 
        cache_path = cache_dict.get_cache_dir()
        
        config: ConfigDTO = ConfigDTO(
            data_path=get_data_directory_path(fl_ctx),
            output_path=output_path,
            cache_path=cache_path,
            computation_params=get_computation_parameters(fl_ctx),
            logger=logger,
            site_name=client_name, 
            cache_dict=cache_dict.get_cache_dict()
        )
        
        # Prepare the Shareable object to send the result to other components
        outgoing_shareable = Shareable()
        
        try:
            if task_name == TASK_NAME_LOCAL_CLIENT_STEP1:
                client_result = helpers.perform_task_step1(config)
                cache_dict.update_cache_dict(client_result['cache'])
                outgoing_shareable['result'] = client_result['output']
            
            elif task_name == TASK_NAME_LOCAL_CLIENT_STEP2:
                client_result = helpers.perform_task_step2(shareable, config)
                cache_dict.update_cache_dict(client_result['cache'])
                outgoing_shareable['result'] = client_result['output']
                
            elif task_name == TASK_NAME_LOCAL_CLIENT_STEP3:
                client_result = helpers.perform_task_step3(shareable, config)
                cache_dict.update_cache_dict(client_result['cache'])
                outgoing_shareable['result'] = client_result['output']
            
            elif task_name == TASK_NAME_LOCAL_CLIENT_STEP4:
                helpers.perform_task_step4(shareable, config)
                cache_dict.remove_cache()
                
            else:
                raise ValueError({
                    'message': 'Invalid task Name',
                    'value': task_name
                })
            return outgoing_shareable

        except Exception as err:
            config.logger.error('Exception: ', err)
            raise Exception(f'exception: {err}')
        finally:
            logger.close()