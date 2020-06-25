from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704

RESTING_BRAKE_Nm = 700


class Controller(object):
    def __init__(self, update_rate_hz, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
                       accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel, max_steer_angle):
                       
        self.update_rate_hz = update_rate_hz
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius
        self.fuel_capacity=fuel_capacity
        self.brake_deadband=brake_deadband
        self.decel_limit=decel_limit
        self.accel_limit=accel_limit
        self.wheel_radius=wheel_radius
        self.wheel_base=wheel_base
        self.steer_ratio=steer_ratio
        self.max_lat_accel=max_lat_accel
        self.max_steer_angle=max_steer_angle
        
        # THROTTLE
        kp = 0.3 # proportional
        ki = 0.08 # integral (i.e. bias)
        kd = 0.  # derivative (i.e. low-pass adjustment for kp)
        mn = 0.  # min throttle
        mx = 1.0 # max throttle
        self.throttle_controller = PID(kp, ki, kd, mn, mx)
        
        # STEERING
        self.yaw_controller = YawController(wheel_base, steer_ratio, 0.1, max_lat_accel, max_steer_angle)
        
        # SMOOTH OUT COMMANDED VELOCITY
        tau = 0.5 # 1/(2pi*tau) = cutoff frequency
        ts = 1.0/update_rate_hz # 0.02 # sample time
        self.vel_lpf = LowPassFilter(tau, ts)
        
        self.last_time = rospy.get_time()

    def control(self, current_vel, dbw_enabled, linear_vel, angular_vel):
        # TODO: Change the arg, kwarg list to suit your needs
        
        if not dbw_enabled:
            self.throttle_controller.reset() # Start it over from scratch (should we keep the integral??)
            return 0.,0.,0. # return nothingness
            
        current_vel = self.vel_lpf.filt(current_vel)
        
        # Get steering and slow down if we're going to fast to make the turn
        steering, new_vel = self.yaw_controller.get_steering(linear_vel,angular_vel,current_vel)
        #rospy.loginfo ("Linear Vel = %s  Current Vel = %s  New Vel = %s" % (linear_vel,current_vel,new_vel))
        overshoot_error = current_vel - new_vel # will be positive if going too fast on turn

        # penalize velocity error, but not if we need to slow down for the turn
        vel_error = linear_vel - current_vel - overshoot_error 
        self.last_vel = current_vel
        
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        
        throttle = self.throttle_controller.step(vel_error, sample_time)
        brake = 0.
        
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0.
            brake = RESTING_BRAKE_Nm
            
        #elif throttle < .1 and vel_error < 0:
        elif vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel) * self.vehicle_mass * self.wheel_radius
            if linear_vel == 0.:
                brake = max(RESTING_BRAKE_Nm,brake)
            #brake += RESTING_BRAKE_Nm
        
        #rospy.loginfo ("Throttle = %s  Brake = %s  Steering = %s" % (throttle,brake,steering))
        return throttle, brake, steering
            
        
        # Return throttle, brake, steer
        #return 0.5, 0., 0.
        return 0., RESTING_BRAKE_Nm, 0.
        #return 0., 0., 0.

