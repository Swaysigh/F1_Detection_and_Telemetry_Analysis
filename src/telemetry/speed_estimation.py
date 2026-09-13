"""Estimate car speed from tracked world-coordinate positions over time."""


def estimate_speed(positions, timestamps):
    """
    positions: list of (x, y) world coords in meters, per frame
    timestamps: list of frame timestamps in seconds
    Returns speed in km/h for each interval.
    """
    speeds = []
    for i in range(1, len(positions)):
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        dist_m = (dx ** 2 + dy ** 2) ** 0.5
        dt = timestamps[i] - timestamps[i - 1]
        if dt > 0:
            speed_mps = dist_m / dt
            speeds.append(speed_mps * 3.6)  # convert to km/h
    return speeds
