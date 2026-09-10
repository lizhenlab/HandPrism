"""Compatibility exports; production identities come from architecture_spec."""

from .architectures import CHECKPOINT_FORMAT, CONTRACT_VERSION, FUSION, architecture_spec

IMPLEMENTATION_ID = architecture_spec(FUSION).implementation_id

__all__ = ["IMPLEMENTATION_ID", "CONTRACT_VERSION", "CHECKPOINT_FORMAT"]
