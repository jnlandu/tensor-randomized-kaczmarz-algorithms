"""
Tensor Randomized Kaczmarz Algorithms

A Python package for solving tensor equations using randomized Kaczmarz methods.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _dist_version


def _get_dist_version() -> str:
	# Distribution name (what you `pip install`)
	for dist_name in ("tensor_toolbox", "tensor-tools"):
		try:
			return _dist_version(dist_name)
		except PackageNotFoundError:
			continue
	return "0.0.0"


class _Version(str):
	def __call__(self) -> str:  # allows `pkg.__version__()`
		return str(self)


__version__ = _Version(_get_dist_version())

__author__ = "Jeremie Mabiala"
__email__ = "jnlandu@gmail.com"


