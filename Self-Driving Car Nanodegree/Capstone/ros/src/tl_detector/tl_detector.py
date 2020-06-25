#!/usr/bin/env python
import rospy
import numpy as np
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import math

STATE_COUNT_THRESHOLD = 3
CLOSEST_STOPLINE_DIST = 0
FARTHEST_STOPLINE_DIST = 80

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        
        
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.waypoints_2d = None
        self.waypoints_tree = None
        #self.light_waypoints_2d = None
        #self.light_waypoints_tree = None
        
        self.light_counter = 0
        self.last_light_dist = 0
        self.last_light_state = -1


        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        #sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb) # TODO: consider /image_raw

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        
        self.traffic_light_model_path = None #rospy.get_param("/traffic_light_classifier_path")

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier(self.traffic_light_model_path)
        self.listener = tf.TransformListener()


        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, lane):
        self.waypoints = lane
        self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in lane.waypoints]
        self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        # This will presumably be updated whenever new light data (such as color change) occurs
        self.lights = msg.lights
        self.light_waypoints_2d = [[light.pose.pose.position.x, light.pose.pose.position.y] for light in msg.lights]
        self.light_waypoints_tree = KDTree(self.light_waypoints_2d)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state in (TrafficLight.RED,TrafficLight.YELLOW) else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, x=None, y=None, pose=None):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        
        if None in (self.waypoints_tree,self.waypoints_2d):
            return None
        
        #TODO implement
        which_tree = self.waypoints_tree
        which_waypoints = self.waypoints_2d

        if pose:
            x = pose.position.x
            y = pose.position.y

        closest_idx = which_tree.query([x,y], 1)[1]
        
        closest_coord = which_waypoints[closest_idx]
        prev_coord = which_waypoints[closest_idx-1]
        
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        
        if val > 0:
            closest_idx = (closest_idx + 1) % len(which_waypoints)

        return closest_idx

        return 0

    def distance(self, waypoints, wp1, wp2):
        # meaures distance going around a loop from wp1 to wp2
        if wp2 < wp1:
            step = -1
            wp2 = wp2 - len(waypoints)
        else:
            step = 1
    
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1, step):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


    def get_light_state(self, light, light_dist=0.0):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
        smaller_cv_image_BGR = cv2.resize(cv_image, (0,0), fx=0.25, fy=0.25)

        guess_state = self.light_classifier.get_classification(smaller_cv_image_BGR)
        
        # avoid taking the same shot over and over again, which might bias the training
        if 0: # NOT SAVING TRAINING DATA ANYMORE
            self.light_counter += 1
            light_filename = "/home/student/CarND-Capstone/traindata/train_%05d_%s_%s.png" % (self.light_counter,true_state,light_dist)
            if true_state != self.last_light_state or abs(light_dist - self.last_light_dist) > 0.5:
                cv2.imwrite(light_filename,smaller_cv_image_BGR)
                self.last_light_dist = light_dist
                self.last_light_state = true_state
                rospy.loginfo("SAVING PHOTO %s %s %s" % (light,true_state,light_dist))

        #true_state = self.lights[light].state
        #return true_state, guess_state
        return guess_state

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None
        car_wp = None
        state = TrafficLight.UNKNOWN

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_wp = self.get_closest_waypoint(pose=self.pose.pose)
#                 rospy.loginfo("WAYPOINT_UPDATER Closest Waypoint=%s" % closest_wp)

            if not car_wp is None:
                #TODO find the closest stopline, i.e. visible traffic light (if one exists)
                light = 0
                light_wp = self.get_closest_waypoint(x=stop_line_positions[0][0],y=stop_line_positions[0][1])
                light_dist = abs(self.distance(self.waypoints.waypoints,car_wp,light_wp))
                for i in range(len(stop_line_positions)):
                    wp = self.get_closest_waypoint(x=stop_line_positions[i][0],y=stop_line_positions[i][1])
                    dist = abs(self.distance(self.waypoints.waypoints,car_wp,wp))
                    #rospy.loginfo("TL_DETECTOR Z i=%s  wp=%s  dist=%s" %(i,wp,dist))
                    if dist < light_dist:
                        light = i
                        light_wp = wp
                        light_dist = dist
                #rospy.loginfo("TL_DETECTOR A Closest Light num=%s  wp=%s  dist=%s" %(light,light_wp,light_dist))
                    
                if CLOSEST_STOPLINE_DIST < light_dist < FARTHEST_STOPLINE_DIST:
                    state = self.get_light_state(light,light_dist)
                    #state, guess_state = self.get_light_state(light,light_dist)
                    stopline_xy = stop_line_positions[light]
                    light_wp = self.get_closest_waypoint(x=stopline_xy[0],y=stopline_xy[1])
                    # UNCOMMENT
                    #rospy.loginfo("TL_DETECTOR B Closest Waypoint Idx=%s  Closest Light Idx=%s  State=%s  Guess State=%s  LightWP=%s" %  (car_wp,light,state,guess_state,light_wp))
                    #rospy.loginfo("TL_DETECTOR B Closest Waypoint Idx=%s  Closest Light Idx=%s  State=%s  LightWP=%s" %  (car_wp,light,state,light_wp))
                    return light_wp, state
                    #return light_wp, state
            #self.waypoints = None  # Why is this in the included code??
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
