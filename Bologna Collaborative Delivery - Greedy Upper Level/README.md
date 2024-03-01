**Model Description:**

The original version of the model uses a bilevel optimization (a.k.a. leader-follower) approach to optimize last-mile deliveries in a given city. The leader represents a city or government authority, whose goal is to minimize the total amount of emissions, as well as the number of LM vehicles in the low-emission zones (LEZs) and low-traffic zones (LTZs) of a city. There are two types of followers: first-mile followers, whose goals are to find the minimum routing cost required to deliver packages to micro-hubs, and last-mile followers, who wish to pick up packages from micro-hubs and deliver them to customers within specific time windows. While a follower might wish to use as many vehicles as possible to ensure packages do not arrive later than expected, the leader wants to ensure that only a minimal number of vehicles are present in the city’s LTZ. In our model, the leader minimizes emissions by first assigning packages from first-milers to micro-hubs, and then from micro-hubs to last-milers in such a way the total amount of emissions and delivery vehicles are minimized.

In this version of the model, the upper-level optimization problem has been removed to improve scalability. The assignment of packages to micro-hubs, as well as the assignment of packages to last-milers is done using a heuristic algorithm. The individual follower problems can then bee solved independently. The first-mile followers each solve a vehicle routing problem with time windows (VRPTW), while the last-mile followers solve a pickup-and-delivery time-dependent vehicle routing problem with time windows (PDTDVRPTW).

**Input files and parameters:**

The input files which give the following information about the problem:

1.  Battery capacity of last-mile electricity delivery vehicles (EDVs)
2.  electricity generation breakdown of the city
3.  number of last-milers and last-mile EDVs available
4.  locker node numbers and capacities
5.  problem information containing node id, type of node, package id, latitude and longitude of node, and delivery time window (earliest and latest times)
6.  time limit for solving each last-mile follower problem.

**Outputs:**

For a given problem instance, the output gives an excel file with three sheets:

1.  Total distance travelled by each last-miler, and the corresponding emissions (gCO2Eq) that occurred during the trip.
2.  The assignments of the parcels to lockers
3.  The arrival time of each package, and the ID of the last-miler who delivered the package.
