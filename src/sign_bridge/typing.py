"""Shared typing aliases for SignBridge.

``NDArray`` is the project-wide alias for the float arrays that flow through
the landmark pipeline. Using one alias keeps signatures readable while still
satisfying ``mypy --strict`` (which requires type parameters on
``numpy.ndarray``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

#: A NumPy array (landmark features, windows, probabilities, image frames).
#: Kept dtype-agnostic so it can annotate both float feature arrays and
#: uint8 BGR video frames without churn.
NDArray = npt.NDArray[Any]

#: A NumPy array of integers (label vectors, indices).
NDArrayInt = npt.NDArray[np.integer[Any]]
