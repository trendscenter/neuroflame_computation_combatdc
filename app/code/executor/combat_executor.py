import logging
import json
import os

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from utils.utils import get_data_directory_path, get_output_directory_path
from utils.task_constants import *

from . import client_cache_store as cache

from . import client_executor_methods as helpers

class DCCombatExecutor(Executor):
    def __init__(self):
        """
        Initialize the SrrExecutor. This constructor sets up the logger.
        """
        logging.info("DCCombat Executor initialized")
        
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
        cache_store = cache.CacheSerialStore(get_output_directory_path(fl_ctx))
        # Prepare the Shareable object to send the result to other components
        outgoing_shareable = Shareable()
        
        if task_name == TASK_NAME_LOCAL_CLIENT_STEP1:
            client_result = helpers.perform_task_step1(fl_ctx)
            cache_store.update_cache_dict(client_result['cache'])
            outgoing_shareable['result'] = client_result['output']
        
        return outgoing_shareable