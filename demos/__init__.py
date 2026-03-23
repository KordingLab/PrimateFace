"""PrimateFace demonstration scripts and notebooks.

Core functionality has moved to the ``primateface`` package::

    import primateface
    pf = primateface.PrimateFace()
    faces = pf.analyze("image.jpg")

For low-level access to the processor, smoother, or visualizer::

    from primateface._processor import PrimateFaceProcessor
    from primateface._smooth import MedianSavgolSmoother
    from primateface._viz import FastPoseVisualizer
"""
