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

import shutil
# import matplotlib.pyplot as plt
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

input_dir_train_iris = ['/home/IrisCode/BIOMDATA_NormalizedImages_jpeg']
input_dir_train_mask = ['/home/IrisCode/BIOMDATA_NormalizedMasks_jpeg']
input_dir_train_code = ['/home/IrisCode/BIOMDATA_IrisCodes_jpeg']

input_dir_test_iris = ['/home/IrisCode/BIOMDATA_NormalizedImages_jpeg']
input_dir_test_mask = ['/home/IrisCode/BIOMDATA_NormalizedMasks_jpeg']
input_dir_test_code = ['/home/IrisCode/BIOMDATA_IrisCodes_jpeg']


# CHKPNT_name_singlemodality=['CHKPNT_iris']


save_dir = 'home/IrisCode/BIOMDATA_chk_threshold_test'

log_dir =save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)



lblfileaddress_train = "/home/IrisCode/osiris/train_2splits"
lblfileaddress_test ='/home/IrisCode/osiris/osiris/test_2splits'

# mode of run

_Mode=0
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
N=['gg_19_joint','gg_19_quality']
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

def read_images_from_disk_triplet(input_queue):
    """Consumes a single filename and label as a ' '-delimited string.
    Args:
      filename_and_label_tensor: A scalar string tensor.
    Returns:
      Two tensors: the decoded image, and the string label.
    """
    # file_contents = np.load(input_queue[0])

    # file_contents = tf.read_file(input_queue[0])
    # example = tf.image.decode_jpeg(file_contents, channels=3)


    file_contents = tf.read_file(input_queue[0])
    example_1 = tf.image.decode_jpeg(file_contents, channels=1)

    file_contents = tf.read_file(input_queue[1])
    example_2 = tf.image.decode_jpeg(file_contents, channels=1)

    file_contents = tf.read_file(input_queue[2])
    example_3 = tf.image.decode_jpeg(file_contents, channels=1)

    # example=tf.reshape(example,samplesize)

    return example_1,example_2,example_3

def load_examples_triplet(datadir, lblfileaddress):
    filename = lblfileaddress
    imgdir = datadir
    # Reads pfathes of images together with their labels
    image_list_1,image_list_2,image_list_3 = read_labeled_image_list_triplet(filename, imgdir)
    # print image_list
    # print label_class_list
    print len(image_list_1)

    images_1 = tf.convert_to_tensor(image_list_1, dtype=tf.string)
    images_2 = tf.convert_to_tensor(image_list_2, dtype=tf.string)
    images_3 = tf.convert_to_tensor(image_list_3, dtype=tf.string)

    # Makes an input queue
    # input_queue = tf.train.slice_input_producer([images, labels],shuffle=isTraining)
    input_queue = tf.train.slice_input_producer([images_1,images_2,images_3], shuffle=True)


    raw_image_1,raw_image_2,raw_image_3 = read_images_from_disk_triplet(input_queue)

    raw_image_1 = tf.image.convert_image_dtype(raw_image_1, dtype=tf.float32)
    raw_image_2 = tf.image.convert_image_dtype(raw_image_2, dtype=tf.float32)
    raw_image_3 = tf.image.convert_image_dtype(raw_image_3, dtype=tf.float32)

    assertion = tf.assert_equal(tf.shape(raw_image_1)[2], n_input_ch, message="image does not have required channels")
    with tf.control_dependencies([assertion]):
        raw_input_1 = tf.identity(raw_image_1)


    assertion = tf.assert_equal(tf.shape(raw_image_2)[2], n_input_ch, message="image does not have required channels")
    with tf.control_dependencies([assertion]):
        raw_input_2 = tf.identity(raw_image_2)

    assertion = tf.assert_equal(tf.shape(raw_image_3)[2], n_input_ch, message="image does not have required channels")
    with tf.control_dependencies([assertion]):
        raw_input_3 = tf.identity(raw_image_3)

    raw_input_1.set_shape([None, None, n_input_ch]) # was 3
    raw_input_2.set_shape([None, None, n_input_ch])  # was 3
    raw_input_3.set_shape([None, None, n_input_ch])  # was 3

    images_1 = preprocess_image_01(raw_input_1, output_height=64, output_width=512, is_training=isTraining)#,resize_side_min=256,resize_side_max=512)
    images_2 = preprocess_image_01(raw_input_2, output_height=64, output_width=512, is_training=isTraining)#,resize_side_min=256,resize_side_max=512)
    images_3 = preprocess_image_01(raw_input_3, output_height=384, output_width=512, is_training=isTraining)#,resize_side_min=256,resize_side_max=512)


    seed = random.randint(0, 2**31 - 1)


    # #scale and crop input image to match 256x256 size
    def transform(image):
        r = image

        return r

    with tf.name_scope("images_1"):
        print images_1.get_shape()

        input_images_1 = transform(images_1)

    with tf.name_scope("images_2"):
        print images_2.get_shape()

        input_images_2 = transform(images_2)

    with tf.name_scope("images_3"):
        print images_3.get_shape()

        input_images_3 = transform(images_3)



    # Optional Image and Label Batching
    image_batch_1,image_batch_2,image_batch_3 = tf.train.batch([input_images_1,input_images_2,input_images_3],
                                              batch_size=batch_size)

    steps_per_epoch = int(math.ceil(len(image_list_1) / batch_size))



    return Examples_triplet(
        images_1=image_batch_1,
        images_2=image_batch_2,
        images_3=image_batch_3,
        count=len(image_list_1),
        steps_per_epoch=steps_per_epoch,
    )


# def create_model_osiris(examples,isTraining):
#
#
#     with tf.name_scope("vgg_anchor"):
#         with tf.variable_scope("vgg_19_iris"):
#             Pred = Uent(examples.images_1,examples.images_2,is_training=isTraining)
#
#
#
#
#     with tf.variable_scope("distance"):
#         Pred=tf.squeeze(Pred)
#
#         distance = tf.reduce_sum(tf.pow(tf.subtract(Pred, examples.images_3), 2),1)
#
#     return distance


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


def create_model(examples,isTraining):
    B = examples.images_2
    B =tf.cast (tf.greater(B,0.5), dtype=tf.float32)

    inputs=tf.concat([examples.images_1,B],3)

    tar=examples.images_3
    tar = tf.cast(tf.greater(tar, 0.5), dtype=tf.float32)
    targets=tf.concat([tar[:,0:64,:,:],tar[:,64:128,:,:],tar[:,128:64*3,:,:],tar[:,64*3:64*4,:,:],tar[:,64*4:64*5,:,:],tar[:,64*5:64*6,:,:]],3)


    with tf.variable_scope("generator"):
        out_channels = int(targets.get_shape()[-1])
        outputs = create_generator(inputs, out_channels,isTraining)

    # create two copies of discriminator, one for real pairs and one for fake pairs
    # they share the same underlying variables

    with tf.name_scope("generator_loss"):
        # predict_fake => 1
        # abs(targets - outputs) => 0
        # gen_loss_GAN = tf.reduce_mean(-tf.log(predict_fake + EPS))
        # outputs=tf.divide((outputs+1.0),2)


        gen_loss_L1 = tf.reduce_mean(tf.abs((2.0*targets-1.0) - outputs))
        Ot=tf.cast(tf.greater(outputs,0),dtype=tf.float32)

        gen_rec=tf.reduce_mean(tf.abs((targets) - Ot))
        O=outputs
        outputs=tf.concat([O[:,:,:,0],O[:,:,:,1],O[:,:,:,2],O[:,:,:,3],O[:,:,:,4],O[:,:,:,5]],1)


        outputs=tf.expand_dims(outputs, -1)
        Ot=tf.cast(tf.greater(outputs,0),dtype=tf.float32)

        # Ot=tf.expand_dims(Ot,-1)



    # with tf.name_scope("generator_train"):
    #     with tf.control_dependencies([discrim_train]):
    #         gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
    #         gen_optim = tf.train.AdamOptimizer(a.lr, a.beta1)
    #         gen_grads_and_vars = gen_optim.compute_gradients(gen_loss, var_list=gen_tvars)
    #         gen_train = gen_optim.apply_gradients(gen_grads_and_vars)

    # ema = tf.train.ExponentialMovingAverage(decay=0.99)
    # update_losses = ema.apply([discrim_loss, gen_loss_GAN, gen_loss_L1])
    #
    # global_step = tf.train.get_or_create_global_step()
    # incr_global_step = tf.assign(global_step, global_step+1)

    return gen_loss_L1,gen_rec,outputs,Ot#Model(
        # predict_real=predict_real,
        # predict_fake=predict_fake,
        # discrim_loss=ema.average(discrim_loss),
        # discrim_grads_and_vars=discrim_grads_and_vars,
        # gen_loss_GAN=ema.average(gen_loss_GAN),
        # gen_loss_L1=ema.average(gen_loss_L1),
        # gen_grads_and_vars=gen_grads_and_vars,
        # outputs=outputs,
        # train=tf.group(update_losses, incr_global_step, gen_train),
    # )
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


# def Wdecay_(S,N,S0):
#     """L2 weight decay loss."""
#     costs = []
#     for var in tf.trainable_variables():
#         if var.op.name.find('gg_19_quality') > 0:
#
#                     print(var.name)
#     sys.exit()
#     return tf.add_n(costs)


def Wdecay_(S,N,S0):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
        if var.op.name.find('weights') > 0:
            f=0
            for c in range(len(S)):
                if var.op.name.find(N[c]) > 0:
                    costs.append(S[c]*tf.nn.l2_loss(var))
                    f=1

            if f==0:
                costs.append(S0 * tf.nn.l2_loss(var))
    return tf.add_n(costs)



def main():
    input_dir_train=[input_dir_train_iris ,input_dir_train_mask ,input_dir_train_code]
    input_dir_test = [input_dir_test_iris, input_dir_test_mask, input_dir_test_code]

    if isTraining:
        lblfileaddress = lblfileaddress_train
        datadir = input_dir_train
        examples = load_examples_triplet(datadir, lblfileaddress)
        num_samples = examples.count
        num_samples_per_epoch = num_samples
        print(">>>>> examples count = %d" % examples.count)
        distance,gen_rec,outputs,Ot = create_model(examples, isTraining)
        distance = tf.reduce_mean(distance)
        gen_rec=tf.reduce_mean(gen_rec)




    else:
        lblfileaddress = lblfileaddress_test
        datadir = input_dir_test
        examples = load_examples_triplet(datadir, lblfileaddress)
        num_samples = examples.count
        num_samples_per_epoch = num_samples
        print(">>>>> examples count = %d" % examples.count)
        distance,gen_rec,outputs,Ot = create_model(examples, isTraining)
        distance = tf.reduce_mean(distance)
        gen_rec=tf.reduce_mean(gen_rec)


    if isTraining:


        with tf.name_scope("loss"):
            #
            # distance = create_model_triplet(examples, isTraining)

            # distance = tf.maximum(distance)

            loss_rec=distance

            loss_reg=Wdecay()*reg_w

            loss_total = loss_reg+loss_rec


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)


    if isTraining:
        with tf.control_dependencies(update_ops):


            global_step = tf.contrib.framework.get_or_create_global_step()
            incr_global_step = tf.assign(global_step, global_step + 1)




            decay_steps = int(num_samples_per_epoch / batch_size * num_epochs_per_decay)
            lr= tf.train.exponential_decay(initial_lr,global_step,decay_steps,learning_rate_decay_factor,staircase=True)

            optim = tf.train.AdamOptimizer(lr, beta1)

            train_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)


            varlistTrain = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            #     if 'fc'  in v.name:
            #         varlistTrain.append(v)


            train_op = optim.minimize(loss_total,var_list=varlistTrain)
            train = tf.group(train_op, incr_global_step)






    if isTraining:
        input_images_1 = examples.images_1
        input_images_2 = examples.images_2
        input_images_3 = examples.images_3


        tf.summary.image("input_1", input_images_1)
        tf.summary.image("input_2", input_images_2)
        tf.summary.image("input_3", input_images_3)
        tf.summary.image("Outputs", outputs)
        tf.summary.image("Ot", Ot)
        tf.summary.image("D", tf.abs(input_images_3-Ot))


        tf.summary.scalar("loss_total", loss_total)
        tf.summary.scalar("loss_reg", loss_reg)
        tf.summary.scalar("loss_rec", loss_rec)



        tf.summary.scalar(("learning rate"),lr)

        # tf.summary.scalar(("q1"), m_plot[0])
        # tf.summary.scalar(("q2"), m_plot[1])
        # tf.summary.scalar(("q3"), m_plot[2])
        #
        # tf.summary.scalar(("q1v"), m_plot_var[0])
        # tf.summary.scalar(("q2v"), m_plot_var[1])
        # tf.summary.scalar(("q3v"), m_plot_var[2])


    # saver1 = tf.train.Saver()

    tf.global_variables_initializer()

    # if initialchkpnt:
    #
    #
    #         varlist = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    #
    #         varlistCHKPNT=[]
    #         for v in varlist:
    #             if 'fc' not in v.name and 'Adam' not in v.name and 'beta1_power' not in v.name and 'beta2_power' not in v.name and 'global_step' not in v.name:
    #                 varlistCHKPNT.append(v)
    #                 print(v.name)
    #
    #
    #         saver1=tf.train.Saver(varlistCHKPNT)




    saver = tf.train.Saver(max_to_keep=1000)

    if isTraining:
        sv = tf.train.Supervisor(logdir=log_dir, save_summaries_secs=120, saver=saver)

    else:
        sv = tf.train.Supervisor()
    # print(CHKPNT_single_modality_folder + '/' + CHKPNT_name_singlemodality[0])

    with sv.managed_session() as sess:

        # if initialchkpnt:
        #     saver1.restore(sess,CHKPNT_single_modality_folder + '/' + CHKPNT_name_singlemodality[0])

        if isLoadModel:

            print ("loading from checkpoint...")
            checkpoint = tf.train.latest_checkpoint(save_dir)
            print ('CHKPNT=',checkpoint)
            # sys.exit()
            saver.restore(sess, checkpoint)


        max_steps = 2**32
        if max_epoch is not None:
            max_steps = examples.steps_per_epoch * max_epoch
            print "max epochs: ", max_epoch
            print "max steps : ", max_steps
            start = time.time()








        if isTraining:
            print('t=1')
            for step in range(max_steps):
                def should(freq):
                    return freq > 0 and ((step + 1) % freq == 0 or step == max_steps - 1)
                fetches = {
                    "train" : train,
                    "global_step" : sv.global_step,
                    "loss" : loss_total,
                    "gen_rec":gen_rec,
                    # "image_1": examples.images_1,
                    # "image_2": examples.images_2,
                    # "image_3": examples.images_3,
                     "lr": lr,
                }



                if should(freq=summary_freq):
                    fetches["summary"] = sv.summary_op
                results = sess.run(fetches)
                print
                print('global_step=',results["global_step"], 'lr=',results["lr"])


                if should(freq=summary_freq):
                    #print("recording summary")
                    sv.summary_writer.add_summary(results["summary"], results["global_step"])
                    # a = np.amin( results["ex"], axis=(1, 2, 3))
                    # b = np.amax( results["ex"], axis=(1, 2, 3))
                    # print results["labels"]
                    # print a.shape
                    # print a
                    # print results["ex"].max(0)
                    #print results["loss"]
                if should(freq=display_step):
                    # global_step will have the correct step count if we resume from a checkpoint
                    train_epoch = math.ceil(results["global_step"] / examples.steps_per_epoch)
                    train_step = (results["global_step"] - 1) % examples.steps_per_epoch + 1
                    rate = (step + 1) * batch_size / (time.time() - start)
                    remaining = (max_steps - step) * batch_size / rate
                    print("progress  epoch %d  step %d  image/sec %0.1f  remaining %dm" % (train_epoch, train_step, rate, remaining / 60))
                    print("loss", results["loss"])
                    print("gen_rec", results["gen_rec"])




                    # A=np.column_stack((results["q1"],results["q2"],results["q3"]))
                    # print ('step:', step, 'A: ', A)
                    # print ('mean=', np.mean(A, axis=0))
                    # print ('var=', np.var(A, axis=0))






                if should(freq=save_freq):
                    print("saving model")
                    # saver = tf.train.Saver(tf.model_variables())
                    saver.save(sess, os.path.join(save_dir, "model"), global_step=sv.global_step)
        else:


            max_steps = examples.count//batch_size

            print "compute distance and labels..."
            t_dist = np.array([])
            t_Ot=np.array([])
            for i in range(max_steps):
                dist = sess.run([distance])
                _gen_rec=sess.run([gen_rec])
                t_dist = np.append(t_dist, dist)
                t_Ot = np.append(t_Ot, _gen_rec)
                print (dist[0])
                print (_gen_rec[0])





main()


