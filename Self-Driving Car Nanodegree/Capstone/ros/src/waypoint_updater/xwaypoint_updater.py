#!/usr/bin/env python
import numpy as np
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, TwistStamped
from styx_msgs.msg import Lane, Waypoint, TrafficLight
from scipy.spatial import KDTree

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
ONE_MPH = 0.44704
MAX_DECEL = 3.0 # m/s^2
MAX_JERK = 10 # m/s^3
#SPEED_LIMIT = 49*ONE_MPH

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # TODO: Add other member variables you need below
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.speed_limit = None
        self.stops_cleared = False
        self.current_vel = None


        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        #rospy.Subscriber('/obstacle_waypoint', Lane, self.obstacle_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)


        #rospy.spin()
        self.loop() # uses rate

    def loop(self):
        rate = rospy.Rate(50) # 30
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                # Get closest waypoint
                #closest_wp = self.get_closest_wp_slow()
                try:
                    closest_wp = self.get_closest_wp_fast()
                    #if closest_wp % 10 == 0:
                    #    rospy.loginfo("WAYPOINT_UPDATER Closest Waypoint=%s" % closest_wp)
                    #if closest_wp == 500:
                    #    self.clear_all_stops()
                    self.publish_waypoints(closest_wp)
                except:
                    pass
            rate.sleep()

    def current_velocity_cb(self,twistStamped):
        self.current_vel = twistStamped.twist.linear.x # how fast are we going now?

    def insert_stop(self,wp_idx):
        if not self.stops_cleared:
            return
        self.stops_cleared = False
        wp_idx -= 5 # Let's stop before we really have to

        # time to accelerate from 0 to V (or vice versa) v/a=t
        starting_speed = self.current_vel
        definitely_stop = False
        if starting_speed < 0.1:
            starting_speed = self.speed_limit
            definitely_stop = True
            
        t_acc = starting_speed / MAX_DECEL
        #t_acc = self.speed_limit / MAX_DECEL
        
        # This is the distance REQUIRED to safely stop
        distance_to_stop = MAX_DECEL/2.0*pow(t_acc,2.) # d = a/2*t^2

        #rospy.loginfo("MAX_DECEL %s  speed_limit %s  t_acc %s  distance to stop %s" % (MAX_DECEL, self.speed_limit,t_acc,distance_to_stop))

        # This is the distance we have AVAILABLE to stop        
        closest_wp = self.get_closest_wp_fast()
        available_distance = abs(self.distance(self.base_waypoints.waypoints,closest_wp,wp_idx))
        
        # Check if we have enough room to stop
        if (not definitely_stop and available_distance < distance_to_stop):
        #if (self.current_vel > self.speed_limit*0.9 and available_distance < distance_to_stop):
            rospy.loginfo("TOO CLOSE TO STOP - WE'RE GOING THROUGH!")
            return

        # Start entering the stop velocities
        rospy.loginfo("Inserting stop at %s" % wp_idx)
        
        for i in range(wp_idx,wp_idx+10):
            self.base_waypoints.waypoints[i].twist.twist.linear.x = 0. # we're stopped at the final point
        
        # Here's a great idea:  let's accelerate backwards from the point!
        idx = wp_idx - 1
        
        d = abs(self.distance(self.base_waypoints.waypoints,idx,wp_idx))
        while d < distance_to_stop:
            #rospy.loginfo("wp_idx %s idx %s  d %s" % (wp_idx,idx,d))
            t = math.sqrt(2*d/MAX_DECEL) # time to travel to that point at acceleration
            v = MAX_DECEL*t
            self.base_waypoints.waypoints[idx].twist.twist.linear.x = v
            #rospy.loginfo("idx %s  a %s  v %s  d %s  t %s" % (idx,MAX_DECEL,v,d,t))
            idx -= 1
            d = abs(self.distance(self.base_waypoints.waypoints,idx,wp_idx))
        
    def set_all_speeds(self,speed):            
        for wp in self.base_waypoints.waypoints:
            wp.twist.twist.linear.x = speed
            
        
    def clear_stops(self):
        if self.speed_limit and not self.stops_cleared:
            self.stops_cleared = True
            rospy.loginfo("Clearing all stops")
            self.set_all_speeds(self.speed_limit)
            
    def pose_cb(self, msg):
        # TODO: Implement
        #rosmsg info geometry_msgs/PoseStamped
        #    std_msgs/Header header
        #      uint32 seq
        #      time stamp
        #      string frame_id
        #    geometry_msgs/Pose pose
        #      geometry_msgs/Point position
        #        float64 x
        #        float64 y
        #        float64 z
        #      geometry_msgs/Quaternion orientation
        #        float64 x
        #        float64 y
        #        float64 z
        #        float64 w
        self.pose = msg
        return

        self.pose_position = msg.pose.position
        self.pose_quaternion = msg.pose.quaternion
        
        # Get closest point to base_waypoints
        closest_waypoint_idx = self.get_closest_pt_slow(msg.pose.position)
        
        # Find the next few points and their velocities
        
        next_few_points = doubledup_waypoints[closest_waypoint_idx:closest_waypoint_idx+LOOKAHEAD_WPS]

        #waypoints = []
        #
        #p = Waypoint()
        #p.pose.pose.position.x = float(wp['x'])
        #p.pose.pose.position.y = float(wp['y'])
        #p.pose.pose.position.z = float(wp['z'])
        #q = self.quaternion_from_yaw(float(wp['yaw']))
        #p.pose.pose.orientation = Quaternion(*q)
        #p.twist.twist.linear.x = float(self.velocity)

        #waypoints.append(p)

        #return self.decelerate(waypoints)
        

        # Publish the next few points to final_waypoints
        self.publish(next_few_points)
        pass

    def publish(self, waypoints):
        lane = Lane()
        lane.header.frame_id = '/world'
        lane.header.stamp = rospy.Time(0)
        lane.waypoints = waypoints
        self.final_waypoints_pub.publish(lane)

#    def distance(self, p1, p2):
#        x, y, z = p1.x - p2.x, p1.y - p2.y, p1.z - p2.z
#        return math.sqrt(x*x + y*y + z*z)

#    def get_closest_wp_slow(self, position=None):
#        if not position:
#            position = self.pose.pose.position
#    
#        closest = 0
#        best_dist = self.distance(position,self.base_waypoints[0].pose.pose.position)
#        for i in range(1,len(self.base_waypoints)):
#            d = self.distance(position,self.base_waypoints[i].pose.pose.position)
#            if (d < best_dist):
#                closest = i
#                best_dist = d
#        return closest

    def get_closest_wp_fast(self,x=None,y=None):
        if None in (x,y):
            x = self.pose.pose.position.x
            y = self.pose.pose.position.y

        closest_idx = self.waypoints_tree.query([x,y], 1)[1]
        
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx-1]
        
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        return closest_idx
        
    def publish_waypoints(self, closest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
        self.final_waypoints_pub.publish(lane)
        

    def waypoints_cb(self, lane):
        # TODO: Implement
        # rosmsg info styx_msgs/Lane
        #    std_msgs/Header header
        #      uint32 seq
        #      time stamp
        #      string frame_id
        #    styx_msgs/Waypoint[] waypoints
        #      geometry_msgs/PoseStamped pose
        #        std_msgs/Header header
        #          uint32 seq
        #          time stamp
        #          string frame_id
        #        geometry_msgs/Pose pose
        #          geometry_msgs/Point position
        #            float64 x
        #            float64 y
        #            float64 z
        #          geometry_msgs/Quaternion orientation
        #            float64 x
        #            float64 y
        #            float64 z
        #            float64 w
        #      geometry_msgs/TwistStamped twist
        #        std_msgs/Header header
        #          uint32 seq
        #          time stamp
        #          string frame_id
        #        geometry_msgs/Twist twist
        #          geometry_msgs/Vector3 linear
        #            float64 x
        #            float64 y
        #            float64 z
        #          geometry_msgs/Vector3 angular
        #            float64 x
        #            float64 y
        #            float64 z
        rospy.loginfo("Got the base waypoints")
        self.base_waypoints = lane # store the waypoints
        
        # waypoint_loader param ~velocity is private, so we're getting it like this - bad ROS, perhaps??
        self.speed_limit = 0.
        for wp in lane.waypoints:
            self.speed_limit = max(self.speed_limit, wp.twist.twist.linear.x)
        #self.insert_stop(500)
        self.clear_stops()       
        self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in lane.waypoints]
        self.waypoints_tree = KDTree(self.waypoints_2d)

        pass

    def traffic_cb(self, wp_idx):
        # TODO: Callback for /traffic_waypoint message. Implement
        if (wp_idx.data >= 0):
            self.insert_stop(wp_idx.data)
        else:
            self.clear_stops()

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
