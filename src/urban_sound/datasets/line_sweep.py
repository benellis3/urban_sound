"""This contains code for performing the line sweep algorithm on a set of intervals
    (https://en.wikipedia.org/wiki/Sweep_line_algorithm). This helps to remove 
    polyphony by removing all start and end times which overlap
"""

from typing import List, NamedTuple


Interval = NamedTuple(
    "Interval", [("is_start", bool), ("index", int), ("value", float)]
)


def remove_overlap(indices: List[int], starts: List[float], ends: List[float]):
    # create the intervals
    intervals = []
    for index, start, end in zip(indices, starts, ends):
        intervals.append(Interval(is_start=True, index=index, value=start))
        intervals.append(Interval(is_start=False, index=index, value=end))

    intervals = sorted(intervals, key=lambda x: x.value)
    current_intervals = set()
    # need to maintain the outermost interval so that we can
    # delete it when the inner intervals are deleted.
    outermost_interval = None
    indices_to_delete = set()
    for interval in intervals:
        if current_intervals and interval.is_start:
            indices_to_delete.add(interval.index)
            # have to check for None explicitly here because the index 0
            # will evaluate to False because bool == int in python :(
            if outermost_interval is not None:
                indices_to_delete.add(outermost_interval)
        if interval.is_start:
            if outermost_interval is None:
                outermost_interval = interval.index
            current_intervals.add(interval.index)
        else:
            current_intervals.remove(interval.index)
            if interval.index == outermost_interval:
                outermost_interval = None

    return list(indices_to_delete)
