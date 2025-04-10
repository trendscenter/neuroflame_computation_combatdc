import os
import numpy as np
import json
import datetime

# Mapping of log levels to numeric values for filtering.
LEVELS = {
    "debug": 10,
    "info": 20,
    "warning": 30,
    "error": 40,
    "critical": 50
}

class NumpyJSONEncoder(json.JSONEncoder):
    """
    Custom JSONEncoder that converts non-JSON-serializable objects into
    JSON-friendly types. Supports:
      - NumPy arrays (converted to lists)
      - NumPy scalars (converted to native Python types)
      - Sets (converted to lists)
      - Fallback to string representation for other types
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.complexfloating,)):
            return {"real": obj.real, "imag": obj.imag}
        if isinstance(obj, set):
            return list(obj)
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)

class NvFlareLogger:
    """
    StructuredLogger writes JSON-formatted log messages into a file and filters
    the messages based on a minimum logging level.

    Initialization parameters:
      - filename: path to the log file.
      - level: minimum logging level. Only messages with a level equal to or
               higher than this threshold will be recorded. (Default: "debug")
    
    Logging methods:
      - debug(), info(), warning(), error(), critical()
    
    Usage:
        logger = StructuredLogger("app.log", level="info")
        logger.debug("This is a debug message")     # Will be skipped if level is "info"
        logger.info("Application started", config=my_config)
        logger.error("An error occurred", error=exc_info)
        logger.close()  # Always close the logger when done.
    """
    def __init__(self, filename, filePath, level="debug"):
        self.filename = filename
        self.filePath = os.path.join(filePath, filename)
        self.file = open(self.filePath, "a", encoding="utf-8")
        # Set logging threshold (lower numeric value means more verbose)
        self.level = level.lower()
        self.level_threshold = LEVELS.get(self.level, 10)
    
    def _log(self, level, *args, **kwargs):
        """
        Internal logging method that only writes the log if the provided level
        meets the logging threshold.
        """
        level = level.lower()
        numeric_level = LEVELS.get(level, 10)
        # Filter out messages that are below the threshold.
        if numeric_level < self.level_threshold:
            return
        
        # Build the log record with UTC timestamp and include level info.
        log_record = {
            "timestamp": datetime.datetime.now().astimezone().strftime("%m/%d/%Y %H:%M:%S") + " : ",
            "level": level,
            "args": args,
            "kwargs": kwargs
        }
        
        try:
            log_line = json.dumps(log_record, cls=NumpyJSONEncoder)
        except Exception as e:
            log_line = json.dumps({
                "timestamp": log_record.get("timestamp"),
                "level": "error",
                "error": str(e)
            })
        
        # Write the log record to file as a new line.
        self.file.write(log_line + "\n")
        self.file.flush()  # Force immediate disk write

    # Public methods for the various log levels:
    def debug(self, *args, **kwargs):
        """Logs a debug level message."""
        self._log("debug", *args, **kwargs)

    def info(self, *args, **kwargs):
        """Logs an info level message."""
        self._log("info", *args, **kwargs)

    def warning(self, *args, **kwargs):
        """Logs a warning level message."""
        self._log("warning", *args, **kwargs)

    def error(self, *args, **kwargs):
        """Logs an error level message."""
        self._log("error", *args, **kwargs)

    def critical(self, *args, **kwargs):
        """Logs a critical level message."""
        self._log("critical", *args, **kwargs)

    def close(self):
        """Closes the log file."""
        if self.file:
            self.file.close()
