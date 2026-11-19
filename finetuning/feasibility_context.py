"""Plan nested spatial-context probes without moving the scored physical cells."""


def plan_context_windows(channels):
    """Use a fixed central two-channel core and stride-16-aligned crop origins.

    All widths fit inside the retained tile; missing spatial context is never
    manufactured. The native encoder still applies its usual internal padding.
    Narrow tiles provide fewer contrasts, rather than being dropped.
    """
    if type(channels) is not int or channels < 1:
        raise ValueError("Expected a positive integer channel count")
    core_width = min(2, channels)
    if channels < 8:
        core_start = (channels - core_width) // 2
    else:
        starts = [
            start for start in range(0, channels - 8 + 1, 16)
            if all(any(origin <= start + 3 and start + 5 <= origin + width
                       for origin in range(0, channels - width + 1, 16))
                   for width in {min(value, channels) for value in (16, 32)})
        ]
        anchor = min(starts, key=lambda start: (abs(start + 4 - channels / 2), start))
        core_start = anchor + 3
    windows = []
    for width in sorted({min(width, channels) for width in (8, 16, 32, channels)}):
        starts = [
            start for start in range(0, channels - width + 1, 16)
            if start <= core_start and core_start + core_width <= start + width
        ]
        if not starts:
            raise ValueError("No aligned context can contain the fixed scoring core")
        start = min(starts, key=lambda value: (
            abs(value + width / 2 - (core_start + core_width / 2)), value))
        windows.append({
            "start": start, "stop": start + width, "channels": width,
            "score_start_in_window": core_start - start,
            "score_stop_in_window": core_start + core_width - start,
            "full_tile": width == channels,
        })
    return {
        "source_channels": channels, "core_start": core_start,
        "core_stop": core_start + core_width, "windows": windows,
        "has_context_contrast": len(windows) > 1,
    }
