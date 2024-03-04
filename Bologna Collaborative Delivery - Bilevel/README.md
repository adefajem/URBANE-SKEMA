**Model Description:**

This model uses a bilevel optimization (a.k.a. leader-follower) approach to optimize last-mile deliveries in a given city. The leader represents a city or government authority, whose goal is to minimize the total amount of emissions, as well as the number of LM vehicles in the low-emission zones (LEZs) and low-traffic zones (LTZs) of a city. There are two types of followers: first-mile followers, whose goals are to find the minimum routing cost required to deliver packages to micro-hubs, and last-mile followers, who wish to pick up packages from micro-hubs and deliver them to customers within specific time windows. While a follower might wish to use as many vehicles as possible to ensure packages do not arrive later than expected, the leader wants to ensure that only a minimal number of vehicles are present in the city’s LTZ. In our model, the leader minimizes emissions by first assigning packages from first-milers to micro-hubs, and then from micro-hubs to last-milers in such a way the total amount of emissions and delivery vehicles are minimized.

In this version of the model, the full bilevel model is solved by creating a single-level relaxation and optimizing this relaxed problem over several iterations. Solving the relaxed problem gives a lower bound (LB). After each iteration, if the follower problems are not optimal, one or more constraints (or cuts) are added, and the relaxed problem is re-solved. These cuts are based on the followers’ objective functions. The combination of follower best-responses is used to calculate an upper bound (UB). The first-mile followers each solve a vehicle routing problem with time windows (VRPTW), while the last-mile followers solve a pickup-and-delivery time-dependent vehicle routing problem with time windows (PDTDVRPTW).

The iteration continues until all follower responses are optimal, or the optimality gap [(UB-LB)/UB] is less than a predefined threshold, or a pre-defined time limit has been exceeded.

**Input files and parameters:**

The problem information files which give the following information about the problem:

1.  Battery capacity of last-mile electricity delivery vehicles (EDVs)
2.  Electricity generation breakdown of the city
3.  First-mile vehicle engine parameters
4.  First-miler information
5.  Last-miler information
6.  Locker capacities
7.  Information on all nodes and package origins
8.  The time-dependent travel times (e.g. distance_times_27.pickle). The distance between any two nodes in the network depending on the time at which the trip takes place.
9.  The latitudes and longitudes of all nodes in the network (e.g. locations_windows_27.pickle), as well as the time windows in which delivery may take place.
10. Time limits for solving the problems.

**Outputs:**

For a given problem instance, the output gives:

1.  Total distance travelled by each last-miler, and the corresponding emissions (gCO2Eq) that occurred during the trip.
2.  The assignments of the parcels to lockers
3.  The arrival time of each package, and the ID of the last-miler who delivered the package.
