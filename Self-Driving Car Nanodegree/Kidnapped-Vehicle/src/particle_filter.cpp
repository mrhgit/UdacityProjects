/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <set>

#include "particle_filter.h"

using namespace std;
#define debug false

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	
	default_random_engine gen; // random number engine class that generates pseudo-random numbers
	
	normal_distribution<double> dist_x(x,std[0]); // Gaussian distributions with mean, std. dev
	normal_distribution<double> dist_y(y,std[1]);
	normal_distribution<double> dist_theta(theta,std[2]);
	
    num_particles = 10; // this gives about 0.1-0.2 meter accuracy.
    particles.resize(num_particles);
    int i = 0;
    for (vector<Particle>::iterator p = particles.begin(); p != particles.end(); ++p){
        (*p).weight = 1.0;
        (*p).x = dist_x(gen);
        (*p).y = dist_y(gen);
        (*p).theta = dist_theta(gen);
        i++;
    }
    
    weights.assign(num_particles,1.0);

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	
	
	// !!!! WE AREN'T GETTING THE ACTUAL STD_MEAS WE SHOULD BE GETTING SENT TO THIS FUNCTION.
	// !!!! IN THIS CASE, I'M JUST GOING TO SHRINK THE GPS MEASUREMENT ERROR DOWN, ASSUMING IT'S
	// !!!! REFLECTIVE OF A FAR LOWER MEASUREMENT ERROR.
	
	
	default_random_engine gen; // random number engine class that generates pseudo-random numbers
	normal_distribution<double> dist_x(0,std_pos[0]/1.); // Gaussian distributions with mean, std. dev
	normal_distribution<double> dist_y(0,std_pos[1]/1.);
	normal_distribution<double> dist_theta(0,std_pos[2]/1.);

	// Update the position of each particle, according to velocity and yaw_rate measurements and delta_t
    // There are two prediction models - one where yaw_rate is 0 and the other where it is not
    if (fabs(yaw_rate) < 0.001){
        double v_delt = velocity*delta_t;
        for (vector<Particle>::iterator p = particles.begin(); p != particles.end(); ++p){
            (*p).x += v_delt * cos((*p).theta) + dist_x(gen);
            (*p).y += v_delt * sin((*p).theta) + dist_y(gen);
            (*p).theta = (*p).theta + dist_theta(gen);
        }
    }
    else {
        //double v_delt_ovyr = velocity*delta_t/yaw_rate;
        double v_ov_yr = velocity/yaw_rate;
        for (vector<Particle>::iterator p = particles.begin(); p != particles.end(); ++p){
            double new_theta = (*p).theta + yaw_rate*delta_t;
            (*p).x += v_ov_yr*(sin(new_theta) - sin((*p).theta)) + dist_x(gen);
            (*p).y += v_ov_yr*(cos((*p).theta) - cos(new_theta)) + dist_y(gen);
            (*p).theta = new_theta + dist_theta(gen);
        }
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

#define pi 3.1415926535
void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	
	// For each particle, we add each observation to the particle's position, then convert to map coordinate system.
	//cout << "updating weights" << endl;
    static double gauss_norm = 1./(2*pi*std_landmark[0]*std_landmark[1]);
	static double x_norm = 2*std_landmark[0]*std_landmark[0];
	static double y_norm = 2*std_landmark[1]*std_landmark[1];
	static double sqr_sensor_range = sensor_range*sensor_range;
	
	weights.clear();
    for (vector<Particle>::iterator p = particles.begin(); p != particles.end(); ++p){
        (*p).weight = 1.0;
        double sin_theta = sin((*p).theta);
        double cos_theta = cos((*p).theta);
        for (vector<LandmarkObs>::const_iterator o = observations.begin(); o != observations.end(); ++o){
            // map-coordinate system version of the observations (made relative to each particle)
            double obs_x_mc = (*p).x + cos_theta*(*o).x - sin_theta*(*o).y;
            double obs_y_mc = (*p).y + sin_theta*(*o).x + cos_theta*(*o).y;
            
            // Look for closest landmark position (non-exclusive)
            double best_sqr_range = sqr_sensor_range + 1.0;
            Map::single_landmark_s *best_landmark = NULL;
            for (vector<Map::single_landmark_s>::const_iterator l = map_landmarks.landmark_list.begin(); l != map_landmarks.landmark_list.end(); ++l){

                // must be within sensor range - more efficient calculation
                double xdiff = fabs((*l).x_f - obs_x_mc); // x distance to landmark from observed point
                if (xdiff < sensor_range){
                    double ydiff = fabs((*l).y_f - obs_y_mc); // y distance to landmark from observed point
                    if (ydiff < sensor_range){
                        double sqr_range = xdiff*xdiff + ydiff*ydiff; // square of range b/t landmark and obs. pt.
                        if ((sqr_range <= sqr_sensor_range) && (sqr_range < best_sqr_range)){
                            best_sqr_range = sqr_range;
                            best_landmark = (Map::single_landmark_s*)&(*l); // gotta love C++ type enforcement...
                        }
                    }
                }

            }
            
            if (best_landmark==NULL) { // nothing is within sensor range
                (*p).weight = 0;
                break; // no point multiplying 0 by any other number in other iterations
            }
            else {
                //cout << "found a landmark for observation" << endl;
                double xdiff = fabs(best_landmark->x_f - obs_x_mc);
                double ydiff = fabs(best_landmark->y_f - obs_y_mc);
                double exponent = xdiff*xdiff/(x_norm) +
                                    ydiff*ydiff/(y_norm);
                double weight = gauss_norm * exp(-exponent);
                (*p).weight *= weight;
            }
        }
        weights.push_back((*p).weight);
    }
    
    //cout << "updating weights complete" << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	// This implementation using the Resampling Wheel approach.
	// First, find the maximum weight, wmax.  Then generate N random integers between 0 and N and N random floats between 0 and 2*wmax.
	// Starting at the random integer and advancing by the random float, end up at the next sample to be chosen.
	if (true) {
	    default_random_engine gen;

	    // Find maximum
	    double wmax = *max_element(weights.begin(),weights.end());
	
	    int N = weights.size();
	    uniform_int_distribution<int> int_dist(0,N-1);
        uniform_real_distribution<double> real_dist(0,2*wmax);
        
        // Sample with replacement
        std::vector<int> the_chosen_ones(N);
        for(int i = 0; i < N; ++i) {
            int wi = int_dist(gen);
            double beta = real_dist(gen);
            while (weights[wi] < beta) {
                beta -= weights[wi];
                wi = (wi + 1) % N;
           }
           the_chosen_ones[i] = wi;
        }
        std::vector<Particle> particles_copy(particles);
        
        particles.clear();
        for (int i = 0; i < N; i++)
            particles.push_back(particles_copy[the_chosen_ones[i]]);
 
    }
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
