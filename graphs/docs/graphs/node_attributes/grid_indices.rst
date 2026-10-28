##############
 Grid indices
##############

Two node attribute builders encode the position of nodes that lie on a
regular grid in a map projection, as integer ``(row, column)`` pairs.
Nodes that are not placed receive ``(-1, -1)``. Both project the node
coordinates with ``pyproj`` using ``proj4_string`` and only consider the
nodes selected by the optional boolean attribute ``mask_node_attr_name``.

The ``RegularGridIndices`` builder finds the largest complete rectangle of
the grid (no holes) and indexes only its nodes. It is used by the spectral
losses, which need a full rectangle to transform:

.. literalinclude:: ../yaml/attributes_regular_grid_indices.yaml
   :language: yaml

The ``LatticeIndices`` builder places *every* selected node on the lattice,
holes included: the lattice origin is the minimum projected ``x`` and ``y``
of the selected nodes and each node is rounded to its nearest lattice
point. A node further than ``absolute_tolerance`` (in projection units)
from its lattice point, or two nodes in one cell, raise an error. It is
used by ``LatticeNoiseConditioning`` in anemoi-models, which draws and
smooths noise on the data lattice:

.. literalinclude:: ../yaml/attributes_lattice_indices.yaml
   :language: yaml
