#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped
import math

from twist_controller import Controller

'''
You can build this node only after you have built (or partially built) the `waypoint_updater` node.

You will subscribe to `/twist_cmd` message which provides the proposed linear and angular velocities.
You can subscribe to any other message that you find important or refer to the document for list
of messages subscribed to by the reference implementation of this node.

One thing to keep in mind while building this node and the `twist_controller` class is the status
of `dbw_enabled`. While in the simulator, its enabled all the time, in the real car, that will
not be the case. This may cause your PID controller to accumulate error because the car could
temporarily be driven by a human instead of your controller.

We have provided two launch files with this node. Vehicle specific values (like vehicle_mass,
wheel_base) etc should not be altered in these files.

We have also provided some reference implementations for PID controller and other utility classes.
You are free to use them or build your own.

Once you have the proposed throttle, brake, and steer values, publish it on the various publishers
that we have created in the `__init__` function.

'''
#rosmsg info geometry_msgs/TwistStamped 
#    std_msgs/Header header
#      uint32 seq
#      time stamp
#      string frame_id
#    geometry_msgs/Twist twist
#      geometry_msgs/Vector3 linear
#        float64 x
#        float64 y
#        float64 z
#      geometry_msgs/Vector3 angular
#        float64 x
#        float64 y
#        float64 z

#rosmsg info dbw_mkz_msgs/SteeringCmd
#    float32 steering_wheel_angle_cmd
#    float32 steering_wheel_angle_velocity
#    bool enable
#    bool clear
#    bool ignore
#    bool quiet
#    uint8 count

#rosmsg info dbw_mkz_msgs/ThrottleCmd 
#    uint8 CMD_NONE=0
#    uint8 CMD_PEDAL=1
#    uint8 CMD_PERCENT=2
#    float32 pedal_cmd
#    uint8 pedal_cmd_type
#    bool enable
#    bool clear
#    bool ignore
#    uint8 count

#rosmsg info dbw_mkz_msgs/BrakeCmd 
#    uint8 CMD_NONE=0
#    uint8 CMD_PEDAL=1
#    uint8 CMD_PERCENT=2
#    uint8 CMD_TORQUE=3
#    float32 TORQUE_BOO=520
#    float32 TORQUE_MAX=3412
#    float32 pedal_cmd
#    uint8 pedal_cmd_type
#    bool boo_cmd
#    bool enable
#    bool clear
#    bool ignore
#    uint8 count



class DBWNode(object):
    def __init__(self):
        rospy.init_node('dbw_node')

        self.update_rate_hz = 50 # in Hz

        #torque = vehicle_mass * wheel_radius * acceleration

        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35) # Need
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413) # Need
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)

        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd',
                                         SteeringCmd, queue_size=1)
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd',
                                            ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd',
                                         BrakeCmd, queue_size=1)

        self.dbw_enabled = False
        self.current_vel = None
        self.curr_ang_vel = None
        self.linear_vel = None
        self.angular_vel = None
        self.throttle = 0.
        self.steer = 0.
        self.brake = 0.
        
        # TODO: Create `Controller` object
        self.controller = Controller(update_rate_hz=self.update_rate_hz,
                                     vehicle_mass=vehicle_mass,
                                     fuel_capacity=fuel_capacity,
                                     brake_deadband=brake_deadband,
                                     decel_limit=decel_limit,
                                     accel_limit=accel_limit,
                                     wheel_radius=wheel_radius,
                                     wheel_base=wheel_base,
                                     steer_ratio=steer_ratio,
                                     max_lat_accel=max_lat_accel,
                                     max_steer_angle=max_steer_angle)

        # TODO: Subscribe to all the topics you need to
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb)
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb)
        rospy.Subscriber('/vehicle/dbw_enabled',Bool,self.vehicle__dbw_enabled_cb)
        self.loop()

    def current_velocity_cb(self,twistStamped):
        self.current_vel = twistStamped.twist.linear.x # how fast are we going now?

    def twist_cmd_cb(self,twistStamped):
        self.linear_vel = twistStamped.twist.linear.x # forward velocity
        self.angular_vel = twistStamped.twist.angular.z # yaw velocity

    def vehicle__dbw_enabled_cb(self,isenabled):
        self.dbw_enabled = isenabled

    def loop(self):
        rate = rospy.Rate(self.update_rate_hz)
        while not rospy.is_shutdown():
            # TODO: Get predicted throttle, brake, and steering using `twist_controller`
            
            # You should only publish the control commands if dbw is enabled
            if not None in (self.current_vel, self.linear_vel, self.angular_vel):
                self.throttle, self.brake, self.steer = self.controller.control(self.current_vel,
                                                                 self.dbw_enabled,
                                                                 self.linear_vel,
                                                                 self.angular_vel)
                #                                                  <proposed linear velocity>,
                #                                                  <proposed angular velocity>,
                #                                                  <current linear velocity>,
                #                                                  <dbw status>,
                #                                                  <any other argument you need>)
            # if <dbw is enabled>:
            #   self.publish(throttle, brake, steer)
            #throttle = 0.5
            #brake = 0.0
            #steer = 0.0
            
            if self.dbw_enabled:
                self.publish(self.throttle,self.brake,self.steer)
            rate.sleep()

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)


if __name__ == '__main__':
    DBWNode()
