"""Score kagent agent trajectories with LangChain's agentevals."""

from .trajectory import (
    event_stream_to_trajectory,
    split_by_invocation,
    tool_call_names,
)

__all__ = [
    "event_stream_to_trajectory",
    "split_by_invocation",
    "tool_call_names",
    "__version__",
]

__version__ = "0.1.0"
