import logging
import json
import os

from nvflare.apis.executor import Executor
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext
from nvflare.apis.shareable import Shareable
from nvflare.apis.signal import Signal
from _utils.utils import get_data_directory_path, get_output_directory_path
from .local_average import get_local_average_and_count

TASK_NAME_GET_LOCAL_AVERAGE_AND_COUNT = "GET_LOCAL_AVERAGE_AND_COUNT"
TASK_NAME_ACCEPT_GLOBAL_AVERAGE = "ACCEPT_GLOBAL_AVERAGE"


class MyExecutor(Executor):
    def execute(
        self,
        task_name: str,
        shareable: Shareable,
        fl_ctx: FLContext,
        abort_signal: Signal,
    ) -> Shareable:

        logging.info(f"Task Name: {task_name}")

        if task_name == TASK_NAME_GET_LOCAL_AVERAGE_AND_COUNT:
            data = load_data(fl_ctx)
            computation_parameters = get_computation_parameters(fl_ctx)
            decimal_places = computation_parameters["decimal_places"]

            local_average_and_count = get_local_average_and_count(
                data, decimal_places)

            save_results_to_file(local_average_and_count,
                                 "local_average.json", fl_ctx)
            shareable = Shareable()
            shareable["result"] = local_average_and_count
            return shareable

        if task_name == TASK_NAME_ACCEPT_GLOBAL_AVERAGE:
            result = {"global_average": shareable["global_average"]}
            save_results_to_file(result, "global_average.json", fl_ctx)
            return Shareable()


def load_data(fl_ctx: FLContext):
    data_dir_path = get_data_directory_path(fl_ctx)
    data_file_filepath = os.path.join(data_dir_path, "data.json")
    logging.info(f"Loading data from: {data_file_filepath}")
    try:
        with open(data_file_filepath, "r") as file:
            data = json.load(file)
        validate_data_format(data)
        return data
    except FileNotFoundError:
        raise RuntimeError(f"Data file not found at {data_file_filepath}")
    except json.JSONDecodeError:
        raise RuntimeError(f"Invalid JSON format in {data_file_filepath}")


def validate_data_format(data):
    """
    Validates that the data is a list of numbers.

    :param data: The data to validate.
    :raises ValueError: If the data is not a list of numbers.
    """
    if not isinstance(data, list) or not all(isinstance(item, (int, float)) for item in data):
        raise ValueError("Data must be a list of numbers")


def get_computation_parameters(fl_ctx: FLContext):
    return fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS", {"decimal_places": 2})


def save_results_to_file(results: dict, file_name: str, fl_ctx: FLContext):
    output_dir = get_output_directory_path(fl_ctx)
    logging.info(f"Saving results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(os.path.join(output_dir, file_name), "w") as f:
            json.dump(results, f, indent=4)
        logging.info(f"Results successfully saved to: {os.path.join(output_dir, file_name)}")
    except Exception as e:
        raise RuntimeError(f"Failed to save results to {file_name}: {e}")
