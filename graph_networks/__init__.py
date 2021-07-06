"""
graph_networks
prediction of chemical properties with graphs and neural networks
"""

# Add imports here
# from .solubility_prediction import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
