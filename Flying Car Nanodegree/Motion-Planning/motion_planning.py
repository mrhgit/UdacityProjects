import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

from planning_utils import a_star_grid, heuristic, create_grid, prune_path_bresenham, graph_from_grid, a_star_graph,convert_graph_path_to_grid
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if len(self.waypoints) > 0:
                close_enough = 4.0
            else:
                close_enough = 1.0
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < close_enough:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # read lat0, lon0 from colliders into floating point values
        with open('colliders.csv') as f:
            [lat0_label, lat0, lon0_label, lon0] = f.readline().replace(",","").split(" ")
        lat0 = float(lat0)
        lon0 = float(lon0)
        print("Read in the lat0 = %f and lon0 = %f" % (lat0,lon0))
        
        # set home position to (lon0, lat0, 0)
        self.set_home_position(lon0, lat0, 0.)

        # retrieve current global position
        global_pos = self.global_position
        print("Global position is ",global_pos)
 
        # convert to current local position using global_to_local()
        local_pos = global_to_local(global_pos, self.global_home)
        print("Local position is ",local_pos)
        
        print('global home {0}, position {1}, local position {2}'.format(self.global_home, self.global_position,
                                                                         self.local_position))
        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)
        
        # Define a grid for a particular altitude and safety margin around obstacles
        grid, north_offset, east_offset = create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        print("North offset = {0}, east offset = {1}".format(north_offset, east_offset))
        # Define starting point on the grid (this is just grid center)
        grid_start = (-north_offset, -east_offset)

        # convert start position to current position rather than map center
        grid_start = (int(local_pos[0]+0.5) - north_offset, int(local_pos[1]+0.5) - east_offset)
        
        # Set goal as some arbitrary position on the grid
        if 0:
            grid_goal = (-north_offset + 10, -east_offset + 10)
        else:
            # adapt to set goal as latitude / longitude position and convert
            #grid_goal_lla = (-122.39801, 37.79257,0.)  # Close by
            grid_goal_lla = (-122.39898, 37.79276,0.)   # Around a first building
            ##grid_goal_lla = (-122.39240, 37.79324,0.)   # Close to the capitol
            #grid_goal_lla = (-122.39793, 37.79202,0.)   # Close to the capitol
            #grid_goal_lla = (-122.39694, 37.79206,0.)   # Close to the capitol
            #grid_goal_lla = (-122.39911, 37.79356,0.)   # Farther Around a first building
            grid_goal = global_to_local(grid_goal_lla, self.global_home)[0:2]
            grid_goal[0] = int(grid_goal[0] - north_offset)
            grid_goal[1] = int(grid_goal[1] - east_offset)
            grid_goal = tuple(np.asarray(grid_goal,dtype='int').tolist())


        # Run A* to find a path from start to goal
        # add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        graph_sizes = [200, 500, 1000, 2000]
        print('Local Start and Goal: ', grid_start, grid_goal)
        for gsize in graph_sizes:
            print("Trying graph size",gsize)
            graph = graph_from_grid(grid,start=grid_start,goal=grid_goal,graph_size=gsize,n_closest=10)
            path, _ = a_star_graph(graph, grid, heuristic, grid_start, grid_goal)
            
            # Use this to plot all the nodes of the search graph
            if 0 and gsize==200:
                waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in graph.nodes]
                self.waypoints = waypoints #np.array(graph.nodes)
                # send waypoints to sim (this is just for visualization of waypoints)
                print(self.waypoints)
                self.send_waypoints()
                time.sleep(5) # Give some time to look at it (and to Ctrl-C)

            if path!=None:
                final_path = convert_graph_path_to_grid(path) # This helps smooth out weird node connections
                break
            # prune path to minimize number of waypoints
        if path==None:
            print("Unable to find that route using my graphs - trying the grid technique.  This may take a minute...")
            path, _ = a_star_grid(grid, heuristic, grid_start, grid_goal)
            final_path = path
            if path==None:
                print ("Can't use that goal!")
                self.waypoints = []
                return
        
        pruned_path = final_path
        do_prune = True
        while(do_prune):
            pruned_path, do_prune = prune_path_bresenham(pruned_path, grid)
        #pruned_path = prune_path_bresenham(pruned_path, grid)
        # (if you're feeling ambitious): Try a different approach altogether!

        if 0:
            # Convert path to waypoints
            waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in final_path]
            # Set self.waypoints
            self.waypoints = waypoints
            # send waypoints to sim (this is just for visualization of waypoints)
            print(self.waypoints)
            self.send_waypoints()
            time.sleep(1)

        waypoints = [[int(p[0] + north_offset), int(p[1] + east_offset), TARGET_ALTITUDE, 0] for p in pruned_path]
        # Set self.waypoints
        self.waypoints = waypoints
        # send waypoints to sim (this is just for visualization of waypoints)
        print(self.waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    drone.start()