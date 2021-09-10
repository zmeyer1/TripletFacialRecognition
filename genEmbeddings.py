#Generate the embedding vectors for a set of images.

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
from keras_vggface.vggface import VGGFace
import sys,os
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

h, w = 224,224
channels_rgb = 3
edims = 128

weights_file = 'VGG_embedding_128.tf'

image_shape = (h, w, channels_rgb)
detector = MTCNN()

def cd(source,file):
    return source+'/'+file

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

#######################################################################
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    try:
        image = Image.open(filename).convert('RGB')
        image = ImageOps.exif_transpose(image)
    except:
        return [],[]
    full_image = np.asarray(image)
    # detect faces in the image
    results = detector.detect_faces(full_image)
    # get bounding box of primary face
    if len(results) == 0:
        return [],[]
    x1, y1, width, height = results[0]['box']
    # ensure everything is positive
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = full_image[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array, full_image

#######################################################################

def main():

    try:
        infolder = sys.argv[1]
    except:
        print("Usage: python3 genEmbeddings.py infolder")
        exit(1)
    try:
        #Open a given directory
        photoFiles = os.listdir(infolder)
    except:
        print("Invalid files")
        exit(1)

    #build model, load weights
    model = build_base_model()
    model.load_weights(weights_file, by_name=False, skip_mismatch=False)
    print("Loaded Weights\n\n")

    count = 0

    #Get each image
    input_im = np.zeros((1,h,w,channels_rgb))
    for i,photoFile in enumerate(photoFiles):
        if not photoFile.lower().endswith(".jpg"):
            continue
        image,full = extract_face(cd(infolder,photoFile))
        if len(image) == 0:
            continue
        input_im[0,:,:,:] = np.asarray(image)
        #Predict embedding
        emb_out = model.predict(input_im)[0]

        #Write to file
        np.savetxt(cd(infolder,photoFile[:-4]+".txt"), emb_out, fmt="%s")
        count+=1
    print("Successfully wrote %d embeddings"%count)



if __name__ == '__main__':
    main()
