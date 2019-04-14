from model import create_model
# model is a program written for the design of our neural network
nn4_small2_pretrained = create_model()
# weights file is for computer to understand what a face looks like and perform operations on it
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')


import numpy as np
import os.path

# A class that serves as a container of image path that the computer understands
class IdentityMetadata():
    def __init__(self, base, name, file):
        # dataset base directory
        self.base = base
        # identity name
        self.name = name
        # image file name
        self.file = file

    def __repr__(self):
        return self.image_path()
    # returns the path to an image
    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

# Function that loads the data that would be used for classification, basically all images present in the 'images' folder
def load_metadata(path):
    metadata = []
    for i in os.listdir(path):
        for f in os.listdir(os.path.join(path, i)):
            # Check file extension. Allow only jpg/jpeg' files.
            ext = os.path.splitext(f)[1]
            if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                metadata.append(IdentityMetadata(path, i, f))
    return np.array(metadata)

metadata = load_metadata('images')


import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from align import AlignDlib
# cv2 for computer vision
# matplotlib for plotting of graphs
# align for aligning face images at a certain angle

######ignore this#####
# %matplotlib inline

# load images at the provided path
def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Initialize the OpenFace face alignment utility to align faces at desired angle
alignment = AlignDlib('models/landmarks.dat')

# Load the first image in our dataset
jc_orig = load_image(metadata[0].image_path())

# Detect face and return bounding box
bb = alignment.getLargestFaceBoundingBox(jc_orig)

#***important comment***#
# Transform image using specified face landmark indices in the initial weights file mentioned above and crop image to only the face (96x96 pixels, required by our model)
jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

####******ignore this******####
# Show original image
# plt.subplot(131)
# plt.imshow(jc_orig)

# Show original image with bounding box
# plt.subplot(132)
# plt.imshow(jc_orig)

# plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

# Show aligned image
# plt.subplot(133)
# plt.imshow(jc_aligned);
####******Till Here******####

# Aligns the image to a straight face
def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# convert image to numerical data
embedded = np.zeros((metadata.shape[0], 128))
# Perform this operation on all images (function names are understandable)
for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    # scale RGB values to interval [0,1]
    try:
        img = (img / 255.).astype(np.float32)
    except:
        continue
    # obtain embedding vector for image (an array of pixel values for that image)
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

# Compute distance between 2 images
def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

# Function to compare 2 image samples
def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));

#show_pair(2, 3)
#show_pair(2, 12)


####****This code is for evaluation of our model (performance evaluation)****####
from sklearn.metrics import f1_score, accuracy_score
distances = []
identical = [] # 1 if same identity, 0 otherwise

num = len(metadata)

for i in range(num - 1):
    for j in range(1, num):
        distances.append(distance(embedded[i], embedded[j]))
        identical.append(1 if metadata[i].name == metadata[j].name else 0)

distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.3, 1.0, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)
# Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
# Accuracy at maximal F1 score
opt_acc = accuracy_score(identical, distances < opt_tau)

# Plot F1 score and accuracy as function of distance threshold
# plt.plot(thresholds, f1_scores, label='F1 score');
# plt.plot(thresholds, acc_scores, label='Accuracy');
# plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
# plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
# plt.xlabel('Distance threshold')
# plt.legend();

dist_pos = distances[identical == 1]
dist_neg = distances[identical == 0]

# plt.figure(figsize=(12,4))
#
# plt.subplot(121)
# plt.hist(dist_pos)
# plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
# plt.title('Distances (pos. pairs)')
# plt.legend();
#
# plt.subplot(122)
# plt.hist(dist_neg)
# plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
# plt.title('Distances (neg. pairs)')
# plt.legend();
####****Till Here****####

####****This code is used to plot the learning done by our model. Basically plots the points representing each image on a graph****####
targets = np.array([m.name for m in metadata])
from sklearn.manifold import TSNE
# sklearn stands for sci-kit learn library in python used for mathematical operations used in machine learning
X_embedded = TSNE(n_components=2).fit_transform(embedded)

for i, t in enumerate(set(targets)):
    idx = targets == t
    plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

plt.legend(bbox_to_anchor=(1, 1));
plt.show()
####****Till Here****####
