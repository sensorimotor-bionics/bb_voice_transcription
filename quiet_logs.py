"""
Silence the third-party log noise that this pipeline cannot act on.

Import this module before torch and nemo. Most of what it suppresses is emitted while
those packages are being imported, so configuring the loggers afterwards would come too
late to stop anything.
"""

import logging
import os

# NeMo reads this when it builds its logger singleton, which happens on the first nemo
# import, so it has to be in the environment before that import rather than set through
# the nemo API afterwards.
os.environ.setdefault("NEMO_LOG_LEVEL", "40")

# Loggers that only ever report on features this pipeline does not use:
#   nv_one_logger        - NVIDIA training telemetry announcing that it is disabled and
#                          has no exporters, which is exactly how we want it.
#   torch.utils.flop_counter - warns that Triton is missing. Triton is only needed to
#                          count FLOPs of Triton kernels, and has no Windows wheel.
_SILENCED_LOGGERS = ("nv_one_logger", "torch.utils.flop_counter")

# NeMo's own logger. Named rather than imported so that this module stays importable
# before nemo is; logging.getLogger returns the same object nemo later picks up.
_NEMO_LOGGER = "nemo_logger"


def _min_level(threshold: int):
    """A logging filter that drops records below threshold."""
    return lambda record: record.levelno >= threshold


def silence_dependency_warnings(threshold: int = logging.ERROR):
    """
    Drop sub-threshold log records from torch, NeMo and NVIDIA's telemetry package.

    Only warnings we can do nothing about are involved, but the suppression is
    per-logger rather than global so that warnings raised by this project, or by any
    package added later, still reach the console.

    Args:
        threshold (int): Lowest level still allowed through, logging.ERROR by default.
    """
    for name in _SILENCED_LOGGERS:
        logging.getLogger(name).setLevel(threshold)

    # NeMo needs a filter rather than a level: diarize() and transcribe() overwrite the
    # verbosity with WARNING for the duration of the call (nemo/collections/asr/parts/
    # mixins/diarization.py) and restore it on the way out, and the Lhotse dataloader
    # and CUDA allocator warnings are raised inside that window. Filters survive
    # setLevel, so the floor asked for here is the floor that holds.
    nemo_logger = logging.getLogger(_NEMO_LOGGER)
    if not any(getattr(f, "_quiet_logs", False) for f in nemo_logger.filters):
        nemo_filter = _min_level(threshold)
        nemo_filter._quiet_logs = True  # makes repeated calls idempotent
        nemo_logger.addFilter(nemo_filter)
