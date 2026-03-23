"""PrimateFace: Cross-species primate face analysis.

Example:
    >>> import primateface
    >>> pf = primateface.PrimateFace()
    >>> faces = pf.analyze("monkey.jpg")
    >>> faces[0].head_pose
    (5.2, -3.1, 1.0)
"""

from .core import PrimateFace
from .face import Face
from . import io

__version__ = "0.2.0"
__all__ = ["PrimateFace", "Face", "io", "__version__"]
