import logging
import numpy as np
import pandas as pd

from typing import Dict, Any, Tuple

from app.code.utils.types import ComputationParamDTO, CombatType
from app.code.utils.logger import NvFlareLogger
from app.code.utils.exceptions import ValidationException


def validate_and_get_inputs(covariates_path: str, data_path: str, combat_algo_type: str, computation_parameters: ComputationParamDTO,
                           logger: NvFlareLogger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        # If given as covariate:datatype as input format
        expected_covariates_info = computation_parameters.get('covariates_types')
        expected_covariates = list(expected_covariates_info.keys())

        # Load the data
        covariates = pd.read_csv(covariates_path)
        data = pd.read_csv(data_path)
        expected_data_info = {key: 'float' for key in data.columns.to_list()}

        # Validate covariates headers
        covariates_headers = set(covariates.columns)
        if not set(expected_covariates).issubset(covariates_headers):
            logger.error("The following covariates are not present in the dataset: {}".format(expected_covariates))
            raise ValidationException("The following covariates are not present in the dataset:",expected_covariates)

        logger.info('Checking data file : ', data_path)

        missing_data_rows = _validate_data_datatypes(data, expected_data_info, logger)
        missing_covariates_rows = _validate_data_datatypes(covariates, expected_covariates_info, logger)

        X, y = None, None

        if len(missing_covariates_rows) > 0:
            raise ValidationException('The following rows are missing from the covariates: ', missing_data_rows)

        if len(missing_data_rows) > 0:
            if combat_algo_type == CombatType.COMBAT_DC.value:
                raise ValidationException('The following rows are having null or empty: ', missing_data_rows)
            elif combat_algo_type == CombatType.COMBAT_MEGA_DC.value:
                if len(missing_covariates_rows) > 0:
                    raise ValidationException('The following rows are missing from the covariates: ', missing_covariates_rows)
            else:
                raise ValidationException('Invalid combat_algo_type: ', combat_algo_type)

        logger.info('converting to required type:')
        X = convert_data_to_given_type(covariates, expected_covariates_info, logger)
        y = convert_data_to_given_type(data, expected_data_info, logger)

        # If all checks pass
        return X, y

    except Exception as e:
        logger.error('exception in validate_and_get_inputs: ', exc_info=e)
        raise e
        # _log_message(error_message, log_path, "error")
        # return False, None, None


def convert_data_to_given_type(data_df: pd.DataFrame, column_info: dict, logger: NvFlareLogger):
    expected_column_names = column_info.keys() # [sex, isControl, age]

    # All the potential
    try:
        for column_name, column_datatype in column_info.items():
            logger.debug(f'Casting datatype of column: ', column_name, ' to the requested datatype: ', column_datatype)
            if column_datatype.strip().lower() == "int":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype(
                    'int')  # or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype('float')
            elif column_datatype.strip().lower() == "str":
                data_df[column_name] = data_df[column_name].astype('object')
            elif column_datatype.strip().lower() == "bool":
                data_df[column_name] = pd.to_numeric(data_df[column_name], errors='coerce').astype('bool')
            else:
                raise Exception(f'Invalid datatype provided in the input for column : {column_name} and datatype: {column_datatype}. Allowed datatypes are int, float, str, bool.')

        # Check for null or NaNs in the converted data
        # curr_rows_to_ignore = data_df[data_df.isnull().any(axis=1)].index.tolist()
        # data_df.drop(data_df.index[curr_rows_to_ignore], inplace=True)

        data_df = data_df[expected_column_names]

    except Exception as e:
        logger.error('An error occurred during data casting: ', exc_info=e)
        raise e

    return data_df


def _validate_data_datatypes(data_df: pd.DataFrame, column_info: dict, logger: NvFlareLogger) -> list:
    all_rows_to_ignore = set()
    try:
        for column_name, column_datatype in column_info.items():
            logger.debug('Validating column: ', column_name, 'with requested datatype: ', column_datatype)
            if column_datatype.strip().lower() == "int":
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('int')  # or .astype('Int64')
            elif column_datatype.strip().lower() == "float":
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('float')
            elif column_datatype.strip().lower() == "str":
                temp = data_df[column_name].astype('object')
            elif column_datatype.strip().lower() == "bool":
                # Converting to int first to make sure all the possible values are converted correctly
                temp = pd.to_numeric(data_df[column_name], errors='coerce').astype('int')  # or .astype('Int64')
            else:
                raise ValidationException(f'Invalid datatype provided in the input for column : {column_name}',
                                f' and datatype: {column_datatype}. Allowed datatypes are int, float, str, bool.')

            # Check for null or NaNs in the data
            null_rows_to_ignore = data_df[temp.isnull()].index.tolist()
            empty_rows_to_ignore = list()
            # Check for emtpy values in the data
            if column_datatype.strip().lower() == "str":
                empty_rows_to_ignore = data_df[temp.str.strip() == ''].index
            logger.debug(f'All rows to ignore: {str(all_rows_to_ignore)}')

            all_rows_to_ignore.update(null_rows_to_ignore, empty_rows_to_ignore)
            if len(null_rows_to_ignore) > 0:
                logger.warning('Ignoring rows with incorrect values for column', column_name, null_rows_to_ignore, empty_rows_to_ignore)
            else:
                logger.warning('Data validation passed for column: ', column_name, ' to the requested datatype', column_datatype)

    except Exception as e:
        logger.error("An error occurred during validation:", exec_info=e)
        raise e

    return list(all_rows_to_ignore)