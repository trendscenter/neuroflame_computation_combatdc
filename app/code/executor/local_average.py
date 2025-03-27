from typing import List, Dict


def get_local_average_and_count(data: List[float], decimal_places: int = 2) -> Dict[str, float]:
    """
    Calculate the average and count from a list of numbers.

    :param data: List of numbers (floats or ints).
    :param decimal_places: Number of decimal places to round the average.
    :return: A dictionary with 'average' and 'count' keys.
    """
    if not data:
        return {"average": 0.0, "count": 0}  # Return default if data is empty

    total_sum = sum(data)
    total_count = len(data)

    average = round(total_sum / total_count, decimal_places)
    return {"average": average, "count": total_count}
