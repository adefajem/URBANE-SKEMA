**Model Description:**

This model solves a Vehicle Routing Problem with Time Windows (VRPTW) in order to determine the optimal route for the delivery robot to deliver goods to a set of customers. The deliveries must occur within specified time windows while minimizing the total distance travelled. This model is an approximation of the actual model used by the LMAD robot. The model uses the CPLEX solver.

**Input Files:**

1.  The robot routes: time taken to cross a link (i.e. go between two nodes) in seconds, the name of the link being traversed, and the length of the link in metres.
2.  The delivery nodes: name of node, type of node, unique node identification number, latitude and longitude
3.  A problem instance which gives the time windows during which the nodes should be served, and the waiting time at each node.

**Input parameters:**

Each problem instance contains a file with the following information.

1.  Node identification number
2.  Earliest time at which a node can be served
3.  Latest time at which a node can be served
4.  The waiting time at each node

**Outputs:**

For the given problem instance the output gives:

1.  The time at which the robot arrives at each hub or delivery point.
2.  The route taken by the robot.
