import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import sys

#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    graph = tf.get_default_graph()

    image_input = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return (image_input, keep_prob, layer3_out, layer4_out, layer7_out)

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
    # set truncated normal weights initializer
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    # set regularizer to prevent weights to explode
    regularizer = tf.contrib.layers.l2_regularizer(scale=1e-3)

    # finish FCN8 architecture based on the VGG model provided
    # here is a best picture which explains architecture: http://blog.playment.io/wp-content/uploads/2018/02/fcn_arch_vgg16.png 
    classes_layer7 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding='same',
                                      kernel_regularizer=regularizer)

    upsampling1 = tf.layers.conv2d_transpose(classes_layer7, num_classes, 4,
                                             strides=(2, 2), padding='same',
                                             kernel_regularizer=regularizer,
                                             kernel_initializer=initializer)


    classes_layer4 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, padding='same',
                                      kernel_regularizer=regularizer)

    upsampling1 = tf.add(upsampling1, classes_layer4)

    upsampling2 = tf.layers.conv2d_transpose(upsampling1, num_classes, 4,
                                             strides=(2, 2), padding='same',
                                             kernel_regularizer=regularizer,
                                             kernel_initializer=initializer)

    classes_layer3 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, padding='same',
                                      kernel_regularizer=regularizer,
                                      kernel_initializer=initializer)

    upsampling2 = tf.add(upsampling2, classes_layer3)

    upsampling3 = tf.layers.conv2d_transpose(upsampling2, num_classes, 16,
                                             strides=(8, 8), padding='same',
                                             kernel_regularizer=regularizer,
                                             kernel_initializer=initializer)
    return upsampling3

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

    entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    cross_entropy_loss = tf.reduce_mean(entropy)

    softmax = tf.nn.softmax(logits)
    iou = tf.metrics.mean_iou(labels, softmax, num_classes)

    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.5

    total_loss = cross_entropy_loss + reg_constant * sum(reg_losses)

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(total_loss)
    #optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(total_loss)

    return (logits, optimizer, cross_entropy_loss, iou)

tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, iou):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.
    :Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """

    print('Started Training......')

    #initialize global variables
    sess.run(tf.global_variables_initializer())

    #initialize local variables (needed for IOU)
    sess.run(tf.local_variables_initializer())

    #training cycle
    for i in range(epochs):
        for (images, labels) in get_batches_fn(batch_size):
            _, training_loss, accuracy = sess.run([train_op, cross_entropy_loss, iou],
                                                  feed_dict={input_image: images,
                                                             correct_label: labels,
                                                             keep_prob: 0.3,
                                                             learning_rate: 0.0002})

        print('Epoch {}'.format(i + 1))
        print('Loss: {0:10.4f} IOU: {1:0.6f}'.format(training_loss, accuracy[0]))

    print('Training Completed.')


tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    # crate placeholders for labels and learning rate
    learning_rate = tf.placeholder(tf.float32)
    correct_label = tf.placeholder(tf.float32,
                                   shape=[None, image_shape[0], image_shape[1], num_classes])

    # number of epochs and batch size,
    # if batch size is greater than 8 - causes memory issues for Video Card
    epochs = 50
    batch_size = 4

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir,
                                                                'data_road/training'), image_shape)

        # Build NN using load_vgg, layers, and optimize function
        (input_image, keep_prob, vgg_layer3_out,
         vgg_layer4_out, vgg_layer7_out) = load_vgg(sess, vgg_path)

        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)

        (logits, train_op, cross_entropy_loss, iou) = optimize(nn_last_layer,
                                                               correct_label,
                                                               learning_rate,
                                                               num_classes)

        # Train NN using the train_nn function
        train_nn(sess, epochs, batch_size, get_batches_fn,
                train_op, cross_entropy_loss, input_image,
                correct_label, keep_prob, learning_rate, iou)

        # save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video

if __name__ == '__main__':
    run()

