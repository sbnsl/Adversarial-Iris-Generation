import  tensorflow as tf
import  numpy as np
import  sys
import glob
import  os
import  random
import math
import collections
import time
import argparse
import cv2

import shutil
import matplotlib.pyplot as plt
########################################################################slim
from tensorflow.contrib import layers
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers as layers_lib
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
slim = tf.contrib.slim
from towers import *
from vgg_preprocessing_10 import *
from rename_singlemodality import *

from depend import calculate_roc_o





parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to folder containing images")
# parser.add_argument("--mode", required=True, choices=["train", "test", "export"])
# parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps (0 to disable)")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100, help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50, help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0, help="trace execution every trace_freq steps")
parser.add_argument("--display_freq", type=int, default=0, help="write current training images every display_freq steps")
parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

parser.add_argument("--separable_conv", action="store_true", help="use separable convolutions in the generator")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--lab_colorization", action="store_true", help="split input image into brightness (A) and color (B)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=286, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--lr", type=float, default=0.0002, help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5, help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0, help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0, help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png", choices=["png", "jpeg"])
a = parser.parse_args()














# Parameter setting ****************************************************************************************************
# directory of input samples

input_dir_train_iris = ['/home/IrisCode/NormalizedImages_jpeg']
input_dir_train_mask = ['/home/IrisCode/NormalizedMasks_jpeg']
input_dir_train_code = ['/home/IrisCode/IrisCodes_jpeg']



input_dir_test_iris = ['/home/IrisCode/NormalizedImages_jpeg']
input_dir_test_mask = ['/home/IrisCode/NormalizedMasks_jpeg']
input_dir_test_code = ['/home/IrisCode/IrisCodes_jpeg']


# CHKPNT_name_singlemodality=['CHKPNT_iris']


save_dir = '/home/IrisCode/chk_fgsm'

log_dir =save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)



lblfileaddress_train = "/home/IrisCode/osiris/train_2splits"
lblfileaddress_test ='/home/IrisCode/osiris/test_2splits'

# mode of run

_Mode=2
mode ="train"
isLoadModel = True

if _Mode==0:
    mode='train'
    isLoadModel = False

elif _Mode==1:
    mode = 'train'
    isLoadModel = True
elif _Mode==2:
    mode='test'
    isLoadModel = True





if mode == "train":
    isTraining = True
else:
    isTraining = False

n_input_ch = 1
batch_size = 64
num_classes=129

reg_w=0.0009*.50
reg_m=10


S=[reg_w*.3,reg_w]
S0=reg_w*.3



# max epoch
max_epoch = 2000

# adam optimizer parameters
beta1 = 0.5 # momentum term of adam

initial_lr = 0.001 # initial learning rate
sync_replicas=0
replicas_to_aggregate=1
num_epochs_per_decay=20
learning_rate_decay_type='exponential'
learning_rate_decay_factor=0.999
end_learning_rate=0.00001



alpha1=.5
alpha2=.5
alpha3=.5



# display parameters

display_step = 20

# saving frequency

save_freq = 1000
summary_freq = 20
samplesize=[300,40,3]



Examples_triplet = collections.namedtuple("Examples", " images_1,images_2,images_3,  count, steps_per_epoch")

def read_labeled_image_list_triplet(image_list_file, img_dir):
    """Reads a .txt file containing pathes and labeles
    Args:
       image_list_file: a .txt file with one /path/to/image per line
       label: optionally, if set label will be pasted after each line
    Returns:
       List with all filenames in file image_list_file
    """
    f = open(image_list_file, 'r')
    filenames_1 = []
    filenames_2 = []
    filenames_3 = []

    for line in f:
        # print line
        filename_1,filename_2,filename_3= line[:-1].split('++')
        # a=img_dir[0][0]
        filenames_1.append(img_dir[0][0]+'/'+filename_1)
        filenames_2.append(img_dir[1][0]+'/'+filename_2)
        filenames_3.append(img_dir[2][0]+'/'+filename_3)


    return filenames_1,filenames_2,filenames_3

def load_data(img_list):
    # a=np.array([cv2.imread( img).flatten() for img in img_list])
    a=np.array([cv2.imread( img,cv2.COLOR_BGR2GRAY) for img in img_list])

    return a

def read_images_from_disk_triplet(input_queue):

    exam=load_data(input_queue)
    return exam

def load_examples_triplet(datadir, lblfileaddress):
    filename = lblfileaddress
    imgdir = datadir
    # Reads pfathes of images together with their labels
    image_list_1,image_list_2,image_list_3 = read_labeled_image_list_triplet(filename, imgdir)
    # print image_list
    # print label_class_list
    print len(image_list_1)

    raw_image_1 = read_images_from_disk_triplet(image_list_1)
    raw_image_2 = read_images_from_disk_triplet(image_list_2)

    raw_image_3 = read_images_from_disk_triplet(image_list_3)

    return raw_image_1,raw_image_2,raw_image_3


#######################################################################################################################################################################################################################################################################################################################################
def gen_conv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        return tf.layers.separable_conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def gen_deconv(batch_input, out_channels):
    # [batch, in_height, in_width, in_channels] => [batch, out_height, out_width, out_channels]
    initializer = tf.random_normal_initializer(0, 0.02)
    if a.separable_conv:
        _b, h, w, _c = batch_input.shape
        resized_input = tf.image.resize_images(batch_input, [h * 2, w * 2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        return tf.layers.separable_conv2d(resized_input, out_channels, kernel_size=4, strides=(1, 1), padding="same", depthwise_initializer=initializer, pointwise_initializer=initializer)
    else:
        return tf.layers.conv2d_transpose(batch_input, out_channels, kernel_size=4, strides=(2, 2), padding="same", kernel_initializer=initializer)


def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)


def batchnorm(inputs,isTraining):
    return tf.layers.batch_normalization(inputs, axis=3, epsilon=1e-5, momentum=0.1, training=isTraining, gamma_initializer=tf.random_normal_initializer(1.0, 0.02))




def create_generator(generator_inputs, generator_outputs_channels,isTraining):
    layers = []

    # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = gen_conv(generator_inputs, a.ngf)
        layers.append(output)

    layer_specs = [
        a.ngf * 2, # encoder_2: [batch, 256, 32, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4, # encoder_3: [batch, 128, 16, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8, # encoder_4: [batch, 64, 8, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8, # encoder_5: [batch, 32, 4, ngf * 8] => [batch, 8, 8, ngf * 8]
        # a.ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        # a.ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        # a.ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]

    for out_channels in layer_specs:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = gen_conv(rectified, out_channels)
            output = batchnorm(convolved,isTraining)
            layers.append(output)

    layer_specs = [
        # (a.ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        # (a.ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        # (a.ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 32, 4, ngf * 8 * 2]
        (a.ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 64, 8, ngf * 4 * 2]
        (a.ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 128, 16, ngf * 2 * 2]
        (a.ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 256, 32, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = gen_deconv(rectified, out_channels)
            output = batchnorm(output,isTraining)
            if isTraining:

                if dropout > 0.0:
                    output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        output = gen_deconv(rectified, generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]


# def create_model_fgsm(isTraining):
#     images_1 = tf.placeholder(tf.float32, shape=(None,64, 512,1))
#     images_2 = tf.placeholder(tf.float32, shape=(None, 64, 512, 1))
#     images_3 = tf.placeholder(tf.float32, shape=(None, 64, 512, 6))
#
#     # B = examples.images_2
#     # B =tf.cast (tf.greater(B,0.5), dtype=tf.float32)
#
#     inputs=tf.concat([images_1,images_2],3)
#
#     tar=images_3
#     tar = tf.cast(tf.greater(tar, 0.5), dtype=tf.float32)
#     targets=tf.concat([tar[:,0:64,:,:],tar[:,64:128,:,:],tar[:,128:64*3,:,:],tar[:,64*3:64*4,:,:],tar[:,64*4:64*5,:,:],tar[:,64*5:64*6,:,:]],3)
#
#
#     with tf.variable_scope("generator"):
#         out_channels = int(targets.get_shape()[-1])
#         outputs = create_generator(inputs, out_channels,isTraining)
#
#     # create two copies of discriminator, one for real pairs and one for fake pairs
#     # they share the same underlying variables
#
#     with tf.name_scope("generator_loss"):
#         # predict_fake => 1
#         # abs(targets - outputs) => 0
#         # gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
#         # outputs=tf.divide((outputs+1.0),2)
#
#
#         gen_loss_L1 = tf.reduce_mean(tf.abs((2.0*targets-1.0) - outputs))
#         Ot=tf.cast(tf.greater(outputs,0),dtype=tf.float32)
#
#         gen_rec=tf.reduce_mean(tf.abs((targets) - Ot))
#         O=outputs
#         outputs=tf.concat([O[:,:,:,0],O[:,:,:,1],O[:,:,:,2],O[:,:,:,3],O[:,:,:,4],O[:,:,:,5]],1)
#
#
#         outputs=tf.expand_dims(outputs, -1)
#         Ot=tf.cast(tf.greater(outputs,0),dtype=tf.float32)
#
#         # Ot=tf.expand_dims(Ot,-1)
#
#
#
#
#     return outputs,Ot
#######################################################################################################################################################################################################################################################################################################################################



def _configure_learning_rate(num_samples_per_epoch, global_step):
    """Configures the learning rate.

    Args:
      num_samples_per_epoch: The number of samples in each epoch of training.
      global_step: The global_step tensor.

    Returns:
      A `Tensor` representing the learning rate.

    Raises:
      ValueError: if
    """
    decay_steps = int(num_samples_per_epoch / batch_size *
                      num_epochs_per_decay)
    if sync_replicas:
        decay_steps /= replicas_to_aggregate

    if learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(initial_lr,
                                          global_step,
                                          decay_steps,
                                          learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
    elif learning_rate_decay_type == 'fixed':
        return tf.constant(initial_lr, name='fixed_learning_rate')
    elif learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(initial_lr,
                                         global_step,
                                         decay_steps,
                                         end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)

def Wdecay():
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
        # if var.op.name.find('weights') > 0:
            costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)





def main():
    input_dir_test = [input_dir_test_iris, input_dir_test_mask, input_dir_test_code]
    lblfileaddress = lblfileaddress_test
    datadir = input_dir_test
    raw_image_1,raw_image_2,raw_image_3 = load_examples_triplet(datadir, lblfileaddress)
    raw_image_1=1.0/255.0*raw_image_1.astype(np.float32)
    raw_image_2 = 1.0/255.0*raw_image_2.astype(np.float32)
    raw_image_3 = 1.0/255.0*raw_image_3.astype(np.float32)

    # raw_image_1=np.transpose(raw_image_1,[0,2,1])
    # raw_image_2 = np.transpose(raw_image_2, [0, 2, 1])
    # raw_image_3 = np.transpose(raw_image_3, [0, 2, 1])

    #######################################
    images_10 = tf.placeholder(tf.float32, shape=(None,64, 512))
    images_20 = tf.placeholder(tf.float32, shape=(None, 64, 512))
    images_30 = tf.placeholder(tf.float32, shape=(None, 384, 512))
    images_1 = tf.expand_dims(images_10, -1)
    images_2 = tf.expand_dims(images_20, -1)
    images_3 = tf.expand_dims(images_30, -1)

    inputs=tf.concat([images_1,images_2],3)

    tar=images_3

    tar = tf.cast(tf.greater(tar, 0.5), dtype=tf.float32)
    targets=tf.concat([tar[:,0:64,:,:],tar[:,64:128,:,:],tar[:,128:64*3,:,:],tar[:,64*3:64*4,:,:],tar[:,64*4:64*5,:,:],tar[:,64*5:64*6,:,:]],3)


    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels,isTraining)


    with tf.name_scope("generator_loss"):

        O=outputs
        outputs_v=tf.concat([O[:,:,:,0],O[:,:,:,1],O[:,:,:,2],O[:,:,:,3],O[:,:,:,4],O[:,:,:,5]],1)
        #

        Ot=tf.cast(tf.greater(outputs,0),dtype=tf.float32)
        Ot_v=tf.cast(tf.greater(outputs_v,0),dtype=tf.float32)


        # predefine cost function for the attack


        print targets.get_shape()
        print outputs.get_shape()
        cost = tf.reduce_mean(tf.abs(targets - outputs))
        # grad = tf.gradient(cost,images_10)[0]






    #######################################

    # load model

    saver = tf.train.Saver()
    sv = tf.train.Supervisor()

    # with tf.Session() as sess:
    with sv.managed_session() as sess:

        print ("loading from checkpoint...")
        checkpoint = tf.train.latest_checkpoint(save_dir)
        print ('CHKPNT=', checkpoint)
        saver.restore(sess, checkpoint)


        r_image_1=raw_image_1[100,:,:]
        r_image_1=np.expand_dims(r_image_1, axis=0)

        r_image_2 = raw_image_2[100, :, :]
        r_image_2 = np.expand_dims(r_image_2, axis=0)

        r_image_3 = raw_image_3[100, :, :]
        r_image_3 = np.expand_dims(r_image_3, axis=0)

        [_output,_Ot,_outputs_v,_Ot_v] = sess.run([outputs,Ot,outputs_v,Ot_v], feed_dict={images_10:r_image_1, images_20:r_image_2, images_30:r_image_3})
        _output=np.squeeze(_output)
        _Ot=np.squeeze(_Ot)

        print(np.mean(np.absolute(_outputs_v-r_image_3)))
        # plt.subplot(221)
        # plt.subplot(2, 1, 1)
        plt.plot(np.squeeze(_outputs_v))
        plt.show()
        # plt.subplot(2, 1, 2)
        plt.plot(np.squeeze(r_image_3))
        plt.show()




        max_attack_it=100



        attacked_img = np.copy(inputimage)

        eps=0.1
        for i in range(max_attack_it):
            [_output, _Ot, grad_, cost_] = sess.run([outputs, Ot, grad, cost],
                                      feed_dict={images_10: attacked_img, images_20: r_image_2, images_30: r_image_3})

            attacked_img = attacked_img  - eps * np.sign(grad_)
            attacked_img = np.clip(attacked_img, a_min=0., a_max=1.)
            print cost_
main()


