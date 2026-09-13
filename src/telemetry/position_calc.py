"""Compute race position (ranking) of cars based on track progress."""


def compute_positions(car_progress):
    """
    car_progress: dict {car_id: distance_along_track_meters}
    Returns list of car_ids sorted by race position (leader first).
    """
    return sorted(car_progress, key=car_progress.get, reverse=True)
