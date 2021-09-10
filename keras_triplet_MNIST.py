import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
import keras.datasets.mnist as mnist
import sys

h, w = 28, 28
edims = 512
channels_rgb = 1 #MNIST is greyscale
triplet_size = 3
nb_classes = 10

image_shape = (h, w, channels_rgb)

def dist(a,b):
    return np.sum(np.square(a-b))

######################################################################

def buildDataSet():
    """Build dataset for train and test


    returns:
        dataset : list of length 10 containing images for each classes of shape (?,28,28,1)
    """
    (x_train_origin, y_train_origin), (x_test_origin, y_test_origin) = mnist.load_data()

    assert K.image_data_format() == 'channels_last'
    x_train_origin = x_train_origin.reshape(x_train_origin.shape[0], h, w, 1)
    x_test_origin = x_test_origin.reshape(x_test_origin.shape[0], h, w, 1)

    dataset_train = []
    dataset_test = []

    #Sorting images by classes and normalize values 0=>1
    for n in range(nb_classes):
        images_class_n = np.asarray([row for idx,row in enumerate(x_train_origin) if y_train_origin[idx]==n])
        dataset_train.append(images_class_n/255)

        images_class_n = np.asarray([row for idx,row in enumerate(x_test_origin) if y_test_origin[idx]==n])
        dataset_test.append(images_class_n/255)

    return dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin

######################################################################

def get_batch_random(batch_size,s="train"):
    """
    Create batch of APN triplets with a complete random strategy
    May be better to use hard triplets

    Arguments:
    batch_size -- integer
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (batch_size,w,h,c)
    """
    if s == 'train':
        X = dataset_train
    else:
        X = dataset_test

    m, w, h,c = X[0].shape

    nb_class = nb_classes-1

    # initialize result
    triplets=[np.zeros((batch_size,h, w,c)) for i in range(3)]

    for i in range(batch_size):
        #Pick one random class for anchor
        anchor_class = np.random.randint(0, nb_class)
        nb_sample_available_for_class_AP = X[anchor_class].shape[0]

        #Pick two different random pics for this class => A and P
        [idx_A,idx_P] = np.random.choice(nb_sample_available_for_class_AP,size=2,replace=False)

        #Pick another class for N, different from anchor_class
        negative_class = (anchor_class + np.random.randint(1,nb_class)) % nb_class
        nb_sample_available_for_class_N = X[negative_class].shape[0]

        #Pick a random pic for this negative class => N
        idx_N = np.random.randint(0, nb_sample_available_for_class_N)

        triplets[0][i,:,:,:] = X[anchor_class][idx_A,:,:,:]
        triplets[1][i,:,:,:] = X[anchor_class][idx_P,:,:,:]
        triplets[2][i,:,:,:] = X[negative_class][idx_N,:,:,:]

    return triplets

def get_batch_hard(draw_batch_size,hard_batchs_size,norm_batchs_size,network,s="train"):
    """
    Create batch of APN "hard" triplets

    Arguments:
    draw_batch_size -- integer : number of initial randomly taken samples
    hard_batchs_size -- integer : select the number of hardest samples to keep
    norm_batchs_size -- integer : number of random samples to add
    Returns:
    triplets -- list containing 3 tensors A,P,N of shape (hard_batchs_size+norm_batchs_size,w,h,c)
    """
    if s == 'train':
        X = dataset_train
    else:
        X = dataset_test

    m, w, h,c = X[0].shape

    #Step 1 : pick a random batch to study
    studybatch = get_batch_random(draw_batch_size,s)

    #Step 2 : compute the loss with current network : d(A,P)-d(A,N). The alpha parameter here is omited here since we want only to order them
    studybatchloss = np.zeros((draw_batch_size))

    #Compute embeddings for anchors, positive and negatives
    A = network.predict(studybatch[0])
    P = network.predict(studybatch[1])
    N = network.predict(studybatch[2])

    #Compute d(A,P)-d(A,N)
    studybatchloss = np.sum(np.square(A-P),axis=1) - np.sum(np.square(A-N),axis=1)

    #Sort by distance (high distance first) and take the
    selection = np.argsort(studybatchloss)[::-1][:hard_batchs_size]

    #Draw other random samples from the batch
    selection2 = np.random.choice(np.delete(np.arange(draw_batch_size),selection),norm_batchs_size,replace=False)

    selection = np.append(selection,selection2)

    triplets = [studybatch[0][selection,:,:,:], studybatch[1][selection,:,:,:], studybatch[2][selection,:,:,:]]

    return triplets

######################################################################

def data_generator(batch_size=32):
    y = np.zeros((batch_size, triplet_size, edims))
    while True:
        x = get_batch_hard(200,batch_size//2,batch_size//2,base_model)
        #yield makes this a generator (keeps the stack)
        yield x,y

######################################################################

def build_base_model():

    inputs = keras.Input(image_shape)

    x = inputs

    x = layers.Conv2D(16, 3, activation="relu")(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.MaxPooling2D(3)(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dense(edims, activation="relu")(x)

    return keras.Model(inputs, x)

######################################################################

base_model = build_base_model()

print('embedding model summary:')
base_model.summary()

img_anchor = keras.Input(image_shape)
img_positive = keras.Input(image_shape)
img_negative = keras.Input(image_shape)

# [None, edims]
y_anchor = base_model(img_anchor)
y_positive = base_model(img_positive)
y_negative = base_model(img_negative)

print(y_anchor)

output_stack = K.stack([y_anchor, y_positive, y_negative], axis=1)

margin = K.constant(0.02)

def triplet_loss(_, y_pred):

    print('y_pred is', y_pred.shape)

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

epochs = 300
batch_size = 32

dataset_train,dataset_test,x_train_origin,y_train_origin,x_test_origin,y_test_origin = buildDataSet()

print("\nDataset_test:",dataset_test[0].shape) #Each Class's shape

test_dataset = np.random.random((triplet_size, 1000, h, w, channels_rgb))
x_prime = [test_dataset[0], test_dataset[1], test_dataset[2]]
embeddings_before_training = triplet_model.predict(x_prime)

triplet_model.fit(data_generator(batch_size),
                steps_per_epoch=epochs,
                 batch_size=batch_size, epochs=1)
base_model.save_weights('triplet_MNIST_weights.tf')
print('saved the weights')

######################################################################

embeddings_after_training = triplet_model.predict(x_prime)

brand_new_model = build_base_model()
brand_new_model.load_weights('triplet_MNIST_weights.tf', by_name=False, skip_mismatch=False)

print('loaded the weights')

for i in range(triplet_size):
    embeddings_single = brand_new_model.predict(x_prime[i])
    embeddings_base = base_model.predict(x_prime[i])
    print('same as embeddings_before_training for', i, '?', np.allclose(embeddings_before_training[:,i], embeddings_single))
    print('same as embeddings_after_training for', i, '?', np.allclose(embeddings_after_training[:,i], embeddings_single))
    print('same as base_model for', i, '?', np.allclose(embeddings_base, embeddings_single))



#Calculate accuracy of Model

refidx = 1
ref_images = np.zeros((nb_classes,h,w,channels_rgb))
for i in range(nb_classes):
    ref_images[i,:,:,:] = dataset_train[i][refidx,:,:,:]
#embeddings for 10 specific training images
ref_embeddings = base_model.predict(ref_images)

print(50*"-")
print("-----------Accuracy Summary-----------")
print(50*"-")

for i in range(nb_classes):
    total = 0
    correct = 0
    images = dataset_test[i][:,:,:,:]
    #embeddings for all the test images
    embeddings = base_model.predict(images)
    min = np.infty
    closest = -1
    for embedding in embeddings:
        for j,ref in enumerate(ref_embeddings):
            distance = dist(embedding,ref)
            if distance <= min:
                min = distance
                closest = j
        if (closest==i):
            correct += 1
        total += 1


    print("-----------Class %d-----------"%i)
    print("Accuracy: %d/%d -- %3.2f"%(correct,total,100*(correct/total)))
