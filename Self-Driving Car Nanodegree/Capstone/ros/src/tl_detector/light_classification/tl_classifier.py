from styx_msgs.msg import TrafficLight
import os
import tensorflow as tf
import numpy as np
import rospy
import cv2

class TLClassifier(object):

    def __init__(self,model_path):
        #TODO load classifier
        self.image_input = None
        self.keep_prob = None
        self.logits = None
        
        self.sess = tf.Session()
        
        self.load_CNN(model_path)
        
        pass

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        return self.cheapoClassifier(image)
        if None in (self.image_input, self.keep_prob, self.logits):
            return TrafficLight.UNKNOWN
        
        label_softmax = self.sess.run([tf.nn.softmax(self.logits)],
                            { self.keep_prob:1.0,
                              self.image_input:[image] })

        label_decision = np.argmax(np.squeeze(label_softmax))
        return label_decision
        
        return TrafficLight.UNKNOWN
        
    def load_CNN(self,save_path, save_name="tl_classifier"):
        if save_path==None:
            return
    
        fullname = os.path.join(save_path, save_name)
        
        rospy.loginfo("Got %s and %s" % (save_path,save_name))
        rospy.loginfo("Loading from %s" % fullname)
        
        loader = tf.train.import_meta_graph(fullname + ".meta")
        loader.restore(self.sess, tf.train.latest_checkpoint(save_path))
        
        graph = self.sess.graph
        
        if (0):
            op = graph.get_operations()
            for m in op:
                print (m.name)
        
        self.image_input = graph.get_tensor_by_name('input_image:0')
        self.keep_prob = graph.get_tensor_by_name('keep_prob:0')
        self.logits = graph.get_tensor_by_name('logits:0')
        return
        
    def cheapoClassifier(self, imgBGR):
        h = 230
        l = 70
        x = 255
        
        red_lights = np.sum(cv2.inRange(imgBGR, (0,0,h), (l,l,x)))
        yel_lights = np.sum(cv2.inRange(imgBGR, (0,h,h), (l,x,x)))
        gre_lights = np.sum(cv2.inRange(imgBGR, (0,h,0), (l,x,l)))
        
        light_ctrs = [red_lights, yel_lights, gre_lights]
        which_light = np.argmax(light_ctrs)
        
        if light_ctrs[which_light]==0:
            return 4
            
        return which_light

