"""Controller registry package.

Importing this package (or anything under it) registers every built-in
controller with `controller_handler._controller_registry`, so
`ControllerFactory.create_controller({"name": "cbf", ...})` works right
after `import irsim_twin` -- no need to separately import each controller
module just for its registration side-effect.
"""

from instance_twin.irsim_borrowed.controller import controller_handler, cbf_qp, c3bf_qp
from instance_twin.irsim_borrowed.controller import local_planners  # noqa: F401
