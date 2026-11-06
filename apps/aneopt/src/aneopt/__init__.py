"""Find and remove Apple Neural Engine bottlenecks in CoreML models. See OPTIMISATION.md for the method.

Submodules import their heavy dependencies themselves, so `aneopt.ioreport` works without torch or coremltools.
"""
__version__ = "0.1.0"
