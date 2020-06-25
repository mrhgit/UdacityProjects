#!/usr/bin/env python3
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import time
import os

# Quiet down, Tensorflow!
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Quiet down, Python!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # Use tf.saved_model.loader.load to load the model and weights
    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = sess.graph

    if (0): # Prints the operations for debug
        op = graph.get_operations()
        for m in op:
            print (m.name)
            print (m.values())

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3 = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4 = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7 = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
    return image_input, keep_prob, layer3, layer4, layer7
print ("\n\nTesting VGG Load...")
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    kernel_size = (2,2)
    strides = (2,2)
    
    layer_up7 = tf.layers.conv2d_transpose(vgg_layer7_out, 512, kernel_size, strides=strides)
    layer_down4_up7 = tf.input = tf.add(layer_up7, vgg_layer4_out)

    layer_up4 = tf.layers.conv2d_transpose(layer_down4_up7, 256, kernel_size, strides=strides)
    layer_down3_up4 = tf.input = tf.add(layer_up4, vgg_layer3_out)

    layer_up3 = tf.layers.conv2d_transpose(layer_down3_up4, num_classes*64, kernel_size, strides=strides)

    layer_up2 = tf.layers.conv2d_transpose(layer_up3, num_classes*16, kernel_size, strides=strides)
    layer_up1 = tf.layers.conv2d_transpose(layer_up2, num_classes, kernel_size, strides=strides)
    
    
    return layer_up1
print ("\n\nTesting Augmented VGG Layers...")
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)
    #train_op = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
print ("\n\nTesting Optimizer...")
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    
    print("\n\nBeginning Training...\n")

    filecount = 289
    batches = round(filecount*1.0/batch_size + 0.5)
    
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        batch_ctr = 0
        tstart = time.time()
        skipped = False
        skipsize = 0
        for batch in get_batches_fn(batch_size):
            (training_images,training_truth) = batch
            
            if training_images.shape[0] < batch_size:
                skipsize = training_images.shape[0]
                skipped = True
                continue

            feed_dict = {input_image: training_images,
                         correct_label: training_truth,
                         learning_rate: 1e-3,
                         keep_prob: 0.75} 
            _, loss = sess.run([train_op,cross_entropy_loss], feed_dict)

            print ("Epoch [%2s/%2s] , Batch [%3s/%3s] , Loss = %f " % (epoch+1,epochs,batch_ctr+1,batches,loss),end="\r")
            batch_ctr += 1
            
        tend = time.time()
        tbatch = tend - tstart
        print ("Epoch [%2s/%2s] , Batch [%3s/%3s] , Loss = %f, Time = %f" % (epoch+1,epochs,batch_ctr,batches,loss,tbatch),end="")
        if skipped:
            print("  (Last Batch of size %d was skipped)" % skipsize,end="")
        print("")


    pass
print ("\n\nTesting Training...")
tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)  # KITTI dataset uses 160x576 images
    data_dir = './data'
    runs_dir = './runs'
    print ("\n\nTesting for KITTI Dataset...")
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    print ("\n\nDownloading Pre-trained model (if don't already have it)...")
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    
    epochs = 25
    batch_size = 8
    keep_prob = tf.placeholder(tf.float32)
    learning_rate = tf.placeholder(tf.float32)
    

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.float32, shape=[None,None,None,num_classes])

        # Build NN using load_vgg, layers, and optimize function
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(layers_output, correct_label, learning_rate, num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image, correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        print("\nSaving Inference Samples...")
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)


        # OPTIONAL: Apply the trained model to a video
    print("\nComplete.")


if __name__ == '__main__':
    run()
