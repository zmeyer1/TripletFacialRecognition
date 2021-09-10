#                                                   Zane Meyer 2021
# Constructs a facial recognition network from
# the VGG face dataset. Uses transfer learning
# to train embeddings using triplet loss
# saves the weights to file


import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
from keras_vggface.vggface import VGGFace
import sys,os
from PIL import Image

h, w = 224,224
edims = 128
channels_rgb = 3 #VGG is color
triplet_size = 3
data_path = '/Volumes/FaceDrive/VGG_50'
classes = os.listdir(data_path)
nb_classes = len(classes)

print("Classes:",nb_classes)

image_shape = (h, w, channels_rgb)


def dist(a,b):
    return np.sum(np.square(a-b))

######################################################################

def get_batch_random(batch_size):
    """
    Create batch of APN triplets with a complete random strategy

    Arguments:
    batch_size -- integer
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """
    c = channels_rgb


    # initialize result
    triplets=[np.zeros((batch_size, h, w, c)) for i in range(triplet_size)]

    for i in range(batch_size):
        #Pick one random class for anchor
        anchor_class = np.random.randint(0, nb_classes)
        anchor_files = os.listdir(data_path + '/' + classes[anchor_class])
        nb_sample_available_for_class_AP = len(anchor_files)

        #Pick two different random pics for this class => A and P
        [idx_A,idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2,replace=False)

        #Pick another class for N, different from anchor_class
        negative_class = (anchor_class + np.random.randint(1,nb_classes)) % nb_classes
        negative_files = os.listdir(data_path + '/' + classes[negative_class])
        nb_sample_available_for_class_N = len(negative_files)

        #Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)
        anchor_path = data_path + '/' + classes[anchor_class] + '/'
        negative_path = data_path + '/' + classes[negative_class] + '/'
        triplets[0][i,:,:,:] = np.asarray(Image.open(anchor_path+anchor_files[idx_A]))
        triplets[1][i,:,:,:] = np.asarray(Image.open(anchor_path+anchor_files[idx_P]))
        triplets[2][i,:,:,:] = np.asarray(Image.open(negative_path+negative_files[idx_N]))

    return triplets

def get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network):
    """
    Create batch of APN "hard" triplets

    Arguments:
    draw_batch_size -- integer : number of initial randomly taken samples
    hard_batchs_size -- integer : select the number of hardest samples to keep
    norm_batchs_size -- integer : number of random samples to add
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """

    #Step 1 : pick a random batch to study
    studybatch = get_batch_random(draw_batch_size)

    #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))

    #Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])

    #Compute d(A,P)-d(A,N)
    studybatchloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)

    #Sort by distance (high distance first) and take the first hard_batch_size triplets (hardest)
    selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]

    #Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)

    selection = np.append(selection,selection2)

    triplets = [studybatch[0][selection,:,:,:], studybatch[1][selection,:,:,:], studybatch[2][selection,:,:,:]]

    return triplets

######################################################################

def data_generator(batch_size=32):
    #Only create this once (save on zeros)
    y = np.zeros((batch_size, triplet_size, edims))
    while True:
        x = get_batch_hard(200,batch_size//2,batch_size//2,base_model)
        #yield makes this a generator (keeps the stack)
        yield x,y

######################################################################

def build_base_model():

    vgg_model = VGGFace(include_top=False, input_shape=image_shape)
    vgg_model.trainable = False
    last_layer = vgg_model.get_layer('pool5').output
    x = layers.MaxPooling2D(pool_size = (3,3),strides = 2)(last_layer)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(edims, activation='relu')(x)
    x = layers.Dense(edims, activation='relu')(x)
    out = layers.Lambda( lambda y: K.l2_normalize(y,axis=-1) )(x)

    return keras.Model(vgg_model.input, out)

######################################################################
weights_file = 'test_weights.tf'

base_model = build_base_model()
#load weights
# base_model.load_weights(weights_file, by_name=False, skip_mismatch=False)

print('embedding model summary:')
base_model.summary()

print("EXITING")
exit(0)

img_anchor = keras.Input(image_shape)
img_positive = keras.Input(image_shape)
img_negative = keras.Input(image_shape)

# [None, edims]
y_anchor = base_model(img_anchor)
y_positive = base_model(img_positive)
y_negative = base_model(img_negative)

print(y_anchor)

output_stack = K.stack([y_anchor, y_positive, y_negative], axis=1)

margin = K.constant(0.2)

def triplet_loss(_, y_pred):

    assert y_pred.shape[1] == triplet_size

    y_anchor = y_pred[:, 0]
    y_positive = y_pred[:, 1]
    y_negative = y_pred[:, 2]

    # each is shape [None]
    dap2 = K.sum(K.square(y_anchor - y_positive), axis=1)
    dan2 = K.sum(K.square(y_anchor - y_negative), axis=1)

    return K.maximum(dap2 - dan2 + margin, 0.0)

triplet_model = keras.Model(inputs=[img_anchor, img_positive, img_negative],
                            outputs=output_stack)

print('\n'*10)
print('triplet model summary:')
triplet_model.summary()

print('\n'*5)
print('compiling!')
triplet_model.compile(loss=triplet_loss)

######################################################################

epochs = 15
batch_size = 32




for i in range(1000):
    triplet_model.fit(data_generator(batch_size),
                    steps_per_epoch=epochs,
                     batch_size=batch_size, epochs=1)
    #Run an error test set.
    print("Fit Completed, calculating loss")
    valid_batch = get_batch_random(200)
    error = triplet_loss(None,triplet_model.predict(valid_batch))
    avg_error = np.sum(error)/len(error)

    error_file = open("error.txt","a")
    error_file.write(str(epochs*(i+1))+" "+str(avg_error)+"\n")
    error_file.close()

    base_model.save_weights(weights_file)
    print('saved the %dth weights'%(i+1))


######################################################################
