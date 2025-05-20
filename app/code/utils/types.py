from typing import NamedTuple, Dict, Any, TypedDict
from enum import Enum

from .logger import NvFlareLogger

class ComputationParamDTO(TypedDict):
  covariate_file: str
  data_file: str
  combat_algo: str
  covariates_types: Dict[str, Any]
  log_level: str

class ConfigDTO(NamedTuple):
  data_path: str
  output_path: str
  cache_path: str
  computation_params: ComputationParamDTO
  cache_dict: Dict[str, Any]
  logger: NvFlareLogger
  site_name: str
  
class CombatType(Enum):
  COMBAT_DC = "combatDC"
  COMBAT_MEGA_DC = "combatMegaDC"