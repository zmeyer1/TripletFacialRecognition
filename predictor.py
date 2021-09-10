#Compare Input image with dataset images

import numpy as np
from silence_tensorflow import auto
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras.backend as K
from keras_vggface.vggface import VGGFace
import sys,os,time
import cv2
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN

h, w = 224,224
channels_rgb = 3
edims = 128

weights_file = 'VGG_embedding_128.tf'

image_shape = (h, w, channels_rgb)
detector = MTCNN()

def dist(a,b):
    return np.sum(np.square(a-b))

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
def extract_face(filename=None,image = None, required_size=(224, 224)):
    # load image from file
    if filename is not None:
        image = Image.open(filename).convert('RGB')
        image = ImageOps.exif_transpose(image)
    elif image is None:
        print("Please give either a filename or image")
        exit(1)
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

def check_closest(distance,label,closest_k,filename,arr_files):
    for i,closest in enumerate(closest_k):
        if closest[0] > distance:
            temp_dist = closest
            temp_file = arr_files[i]
            closest_k[i] = (distance,label)
            arr_files[i] = filename
            check_closest(temp_dist[0],temp_dist[1],closest_k,temp_file,arr_files)
            return

#################

def main():

    data_path = 'faces'
    people = os.listdir(data_path)
    for i,p in enumerate(people):
        if not os.path.isdir(cd(data_path,p)):
            people.pop(i)

    #load model & weights
    model = build_base_model()
    model.load_weights(weights_file, by_name=False, skip_mismatch=False)

    try:
        cpu_flag = False
        cpu_flag = sys.argv[1]
    except:
        pass

    os.system("clear")
    in_str = ""
    while True:

        in_str = input("Please Input an image Filepath: ").lower()
        if in_str in ["quit","exit","q"]:
            break
        elif in_str == "clear":
            os.system("clear")
            continue
        elif in_str == "flag":
            cpu_flag = not cpu_flag
            continue
        elif in_str == "camera":
            pass
        elif not os.path.exists(in_str):
            print("%s does not exist"%in_str)
            continue

        vid_flag = 'file'
        if in_str == 'camera':
            vid_flag = 'camera'
            vc = cv2.VideoCapture(0)
            face = []
            while len(face) == 0:
                rval, frame = vc.read()
                cv2.imshow("Get Face Into View",frame)
                cv2.waitKey(25)
                face,_ = extract_face(image = frame)

            del vc
            cv2.destroyAllWindows()
        else:
            filename = in_str

        if vid_flag != "camera":
            face,_ = extract_face(filename)

        if len(face) == 0:
            print("Could not find a face in the Image")
            exit(0)
        im_in = np.zeros((1,h,w,channels_rgb))
        im_in[0,:,:,:] = face
        emb_in = model.predict(im_in)[0]

        #read in all other embeddings
        k = 5

        num_people = len(people)
        closest_k = [(np.inf,None) for i in range(k)]
        closest_files = ["" for i in range(k)]

        for person in people:
            person_file = cd(data_path,person)
            embed_files = os.listdir(person_file)

            num_embeddings = 0
            for i,embed_file in enumerate(embed_files):
                if embed_file.endswith('.txt'):
                    num_embeddings += 1

            for embed_file in embed_files:
                if not embed_file.endswith('.txt'):
                    continue
                embedding = np.loadtxt(cd(person_file,embed_file))
                #update k-nearest
                distance = dist(embedding,emb_in)
                check_closest(distance,person,closest_k,cd(person_file,embed_file),closest_files)


        cutoff = 0.4
        #Just calculate k-nearest label with cutoff
        label_count = np.zeros(num_people)
        for i,(proximity,name) in enumerate(closest_k):
            if proximity > cutoff:
                closest_k[i] = (proximity,name + " (Not Counted)")
            else:
                label_loc = (1+(k-i)/10)*np.where(np.array(people) == name,1,0)
                label_count = label_count + label_loc

        print(people)
        print(label_count.astype(np.float16))

        if not label_count.any():
            label = "No Match Found"
        else:
            label_idx = np.argmax(label_count)
            label = people[label_idx]

        #Return name, image
        print(label)
        print("Nearest Points: " )
        for i in range(0,k):
            print("\t\t %5.4f (%s)"%(closest_k[i][0],closest_k[i][1]))

        os.system("say "+label)

        if cpu_flag:
            #display image and nearest image to it
            filen = closest_files[0]
            filen = filen[:-4] + ".JPG" #convert txt to jpg
            #get nearest person
            close_im = Image.open(filen)
            close_im = ImageOps.exif_transpose(close_im)
            close_im = np.asarray(close_im.convert('RGB'))
            if vid_flag == "camera":
                current_im = frame
            else:
                im = Image.open(filename).convert('RGB')
                im = ImageOps.exif_transpose(im)
                current_im = np.asarray(im)

            f, (ax1, ax2) = plt.subplots(1, 2)
            ax1.imshow(current_im)
            ax1.set_title("Current")
            ax1.axis('off')
            ax2.imshow(close_im)
            ax2.set_title("Closest")
            ax2.axis('off')
            plt.show()



if __name__ == "__main__":
    main()
