# `x as x` marks public re-exports for type-checkers, see torchcodec/__init__.py.
from ._index_based import (
    clips_at_random_indices as clips_at_random_indices,
    clips_at_regular_indices as clips_at_regular_indices,
)
from ._time_based import (
    clips_at_random_timestamps as clips_at_random_timestamps,
    clips_at_regular_timestamps as clips_at_regular_timestamps,
)
