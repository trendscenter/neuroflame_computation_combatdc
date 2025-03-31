import json
import os
import logging
from nvflare.apis.fl_constant import FLContextKey
from nvflare.apis.fl_context import FLContext

import datetime


def is_repo_root(path: str) -> bool:
    """Check if the given path is the repository root by looking for 'system' and 'app' directories."""
    return all(os.path.isdir(os.path.join(path, d)) for d in ("system", "app"))

def find_repo_root_path() -> str:
    """Find the repository root directory by searching upward for 'system' and 'app' directories."""
    path = os.getcwd()

    while not is_repo_root(path):
        parent = os.path.dirname(path)
        if parent == path:  # Reached filesystem root
            raise FileNotFoundError("Repo root directory could not be found.")
        path = parent

    return path


def get_data_directory_path(fl_ctx: FLContext) -> str:
    """Determine and return the data directory path based on the available paths."""
    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)

    # Check if the environment variable DATA_DIR is set
    env_path = os.getenv("DATA_DIR")
    if env_path and os.path.exists(env_path):
        logging.info(f"Data directory path from environment: {env_path}")
        return env_path

    # If DATA_DIR is not set, use the simulator and poc path
    repo_root_path = find_repo_root_path()
    simulator_and_poc_path = os.path.join(repo_root_path, f"test_data/{site_name}")
    
    if os.path.exists(simulator_and_poc_path):
        logging.info(f"Data directory path for simulator and poc: {simulator_and_poc_path}")
        return simulator_and_poc_path

    raise FileNotFoundError("Data directory path could not be determined.")

def get_output_directory_path(fl_ctx: FLContext) -> str:
    """Determine and return the output directory path based on the available paths."""
    job_id = fl_ctx.get_job_id()
    site_name = fl_ctx.get_prop(FLContextKey.CLIENT_NAME)
    
    # need to check in neuroflame
    if site_name == None:
        site_name = 'remote'

    # Check if the environment variable OUTPUT_DIR is set
    env_path = os.getenv("OUTPUT_DIR")
    if env_path:
        os.makedirs(env_path, exist_ok=True)
        logging.info(f"Output directory path from environment: {env_path}")
        return env_path

    # If OUTPUT_DIR is not set, use the simulator and poc path
    repo_root_path = find_repo_root_path()
    simulator_and_poc_path = os.path.join(repo_root_path, f"test_output/{job_id}/{site_name}")
    os.makedirs(simulator_and_poc_path, exist_ok=True)
    logging.info(f"Output directory path for simulator and poc: {simulator_and_poc_path}")
    return simulator_and_poc_path

def get_parameters_file_path(fl_ctx: FLContext) -> str:
    """Determine and return the parameters file path based on the available paths."""

    # Check if the environment variable PARAMETERS_FILE_PATH is set and the file exists
    env_path = os.getenv("PARAMETERS_FILE_PATH")
    if env_path and os.path.exists(env_path):
        logging.info(f"Parameters file path from environment: {env_path}")
        return env_path

    # If PARAMETERS_FILE_PATH is not set, use the simulator and poc path
    # repo_root_path = find_repo_root_path()
    repo_root_path = find_repo_root_path()
    simulator_and_poc_path = os.path.abspath(os.path.join(repo_root_path, "test_data/server/parameters.json"))
    
    print("simulator_and_poc_path", simulator_and_poc_path)
    
    if os.path.exists(simulator_and_poc_path):
        logging.info(f"Parameters file path for simulator and poc: {simulator_and_poc_path}")
        return simulator_and_poc_path

    raise FileNotFoundError("Parameters file path could not be determined.")

def get_computation_parameters(fl_ctx: FLContext):
    return fl_ctx.get_peer_context().get_prop("COMPUTATION_PARAMETERS", {"decimal_places": 2})

def log(fl_ctx: FLContext, message: str, message_level : str = "info" ) -> None:
    """
    Logs details to log file.

    Args:
        log_path: path of the log file
        message_level: "error" or "info". Default level is info

    Returns:

    """
    log_path = os.path.join(get_output_directory_path(fl_ctx), "client_log.txt")
    time_prefix = datetime.datetime.now().astimezone().strftime("%m/%d/%Y %H:%M:%S") + ' : '
    message = time_prefix + message

    if message_level.strip().lower() == "error":
        logging.error(message)
    else:
        logging.info(message)

    try:
        with open(log_path, 'a') as f:
            f.write(f"{message}\n")
            f.flush()  # Ensure data is written to the file
    except IOError as e:
        logging.error(f"Failed to write to log file {log_path}: {e}")

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
