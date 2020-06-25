import argparse
import time
from enum import Enum

import numpy as np

from udacidrone import Drone
from udacidrone.connection import MavlinkConnection, WebSocketConnection  # noqa: F401
from udacidrone.messaging import MsgID
from queue import Queue

class Phases(Enum):
    MANUAL = 0
    ARMING = 1
    TAKEOFF = 2
    WAYPOINT = 3
    LANDING = 4
    DISARMING = 5


class BackyardFlyer(Drone):

    def __init__(self, connection):
        super().__init__(connection)
        self.target_position = np.array([0.0, 0.0, 0.0])
        self.all_waypoints = Queue()
        self.in_mission = True
        self.check_state = {}
        self.calculate_box()
        self.close_enough = 0.5

        # initial state
        self.flight_phase = Phases.MANUAL # Assume we're starting off in manual/disarmed mode

        # Register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def state_callback(self):
        """
        TODO: Implement this method

        This triggers when `MsgID.STATE` is received and self.armed and self.guided contain new data
        """
        if not self.in_mission:
            return
        if self.flight_phase==Phases.MANUAL: # Let's tap into the controls
            self.arming_transition()
        elif self.flight_phase==Phases.ARMING: # Time to fly
            self.takeoff_transition()
        elif self.flight_phase==Phases.DISARMING: # Fun is over
            self.manual_transition()

    def get_altitude(self): # Returns altitude in a NEU coordinate system (vs NED)
        return -1.0 * self.local_position[2]

    def NEU(self,NED): # Convert NED to NEU
        return [NED[0],NED[1],-1.0*NED[2]]

    def distance_to_waypoint(self):
        return np.sum(np.abs(self.NEU(self.local_position) - self.target_position))

    def command_waypoint(self,waypoint):
        print ("Advancing Waypoint to ",waypoint)
        self.target_position = waypoint
        self.cmd_position(*self.target_position,0.0)

    def local_position_callback(self):
        """
        TODO: Implement this method

        This triggers when `MsgID.LOCAL_POSITION` is received and self.local_position contains new data
        """
        if self.flight_phase ==Phases.TAKEOFF:
            if self.get_altitude() > 0.95 * self.target_position[2]:
                print ("Takeoff Complete")
                self.flight_phase = Phases.WAYPOINT
                self.advance_waypoint()
        elif self.flight_phase==Phases.WAYPOINT:
            if self.distance_to_waypoint() < self.close_enough:
                self.advance_waypoint()
        elif self.flight_phase == Phases.LANDING:
            if ((self.global_position[2] - self.global_home[2] < 0.1) and
                abs(self.local_position[2] < 0.01)):
               print ("Landing Complete")
               self.disarming_transition()


    def velocity_callback(self):
        """
        TODO: Implement this method

        This triggers when `MsgID.LOCAL_VELOCITY` is received and self.local_velocity contains new data
        """
        pass

    def advance_waypoint(self):
        # Check if we're done with our plan
        if self.all_waypoints.empty():
            self.landing_transition()
            return

        self.command_waypoint(self.all_waypoints.get())
        if self.all_waypoints.empty():
            self.close_enough = 0.15 # For our last point, be really precise
        else:
            self.close_enough = 0.75 # otherwise, round the bases

    def calculate_box(self):
        """TODO: Fill out this method
        
        1. Return waypoints to fly a box
        """
        # Points are NEUB (North, East, Up, Bearing)
        #for pt in [(10,10,3),(20,0,3),(10,-10,3),(0,0,3)]: # Technically also a square, but I figured I might fail the review anyway...
        for pt in [(0,10,3),(10,10,3),(10,0,3),(0,0,3)]:
            self.all_waypoints.put(np.array(pt,dtype="float"))
        return

    def arming_transition(self):
        """TODO: Fill out this method
        
        1. Take control of the drone
        2. Pass an arming command
        3. Set the home location to current position
        4. Transition to the ARMING state
        """
        print("arming transition")
        self.take_control()
        self.arm()
        self.set_home_position(*self.global_position[0:3])

        self.flight_phase = Phases.ARMING

    def takeoff_transition(self):
        """TODO: Fill out this method
        
        1. Set target_position altitude to 3.0m
        2. Command a takeoff to 3.0m
        3. Transition to the TAKEOFF state
        """
        print("takeoff transition")
        target_altitude = 3.0
        self.target_position[2] = target_altitude
        self.takeoff(target_altitude)
        self.flight_phase = Phases.TAKEOFF

    def waypoint_transition(self):
        """TODO: Fill out this method
    
        1. Command the next waypoint position
        2. Transition to WAYPOINT state
        """
        print("waypoint transition")

    def landing_transition(self):
        """TODO: Fill out this method
        
        1. Command the drone to land
        2. Transition to the LANDING state
        """
        print("landing transition")
        self.land()
        self.flight_phase = Phases.LANDING

    def disarming_transition(self):
        """TODO: Fill out this method
        
        1. Command the drone to disarm
        2. Transition to the DISARMING state
        """
        print("disarm transition")
        self.disarm()
        self.flight_phase = Phases.DISARMING

    def manual_transition(self):
        """This method is provided
        
        1. Release control of the drone
        2. Stop the connection (and telemetry log)
        3. End the mission
        4. Transition to the MANUAL state
        """
        print("manual transition")

        self.release_control()
        self.stop()
        self.in_mission = False
        self.flight_state = Phases.MANUAL

    def start(self):
        """This method is provided
        
        1. Open a log file
        2. Start the drone connection
        3. Close the log file
        """
        print("Creating log file")
        self.start_log("Logs", "NavLog.txt")
        print("starting connection")
        self.connection.start()
        print("Closing log file")
        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), threaded=False, PX4=False)
    #conn = WebSocketConnection('ws://{0}:{1}'.format(args.host, args.port))
    drone = BackyardFlyer(conn)
    time.sleep(2)
    drone.start()
