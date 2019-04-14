from model import create_model
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')
import pickle
import time
import numpy as np
import os.path
import warnings
warnings.filterwarnings("ignore")
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

    def image_path(self):
        return os.path.join(self.base, self.name, self.file)

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
#import face_recognition
from align import AlignDlib

#%matplotlib inline

def load_image(path):
    img = cv2.imread(path, 1)
    # OpenCV loads images with color channels
    # in BGR order. So we need to reverse them
    return img[...,::-1]

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')

# Load an image of Jacques Chirac
jc_orig = load_image(metadata[0].image_path())

# Detect face and return bounding box

#jc_orig = face_recognition.load_image_file(metadata[2].image_path())
bb = alignment.getLargestFaceBoundingBox(jc_orig)

# Transform image using specified face landmark indices and crop image to 96x96
jc_aligned = alignment.align(96, jc_orig, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

# Show original image
plt.subplot(131)
plt.imshow(jc_orig)

# Show original image with bounding box
plt.subplot(132)
plt.imshow(jc_orig)

#plt.gca().add_patch(patches.Rectangle((bb.left(), bb.top()), bb.width(), bb.height(), fill=False, color='red'))

# Show aligned image
plt.subplot(133)
plt.imshow(jc_aligned);

def align_image(img):
    return alignment.align(96, img, alignment.getLargestFaceBoundingBox(img),
                           landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)

embedded = np.zeros((metadata.shape[0], 128))

for i, m in enumerate(metadata):
    img = load_image(m.image_path())
    img = align_image(img)
    # scale RGB values to interval [0,1]
    try:
        img = (img / 255.).astype(np.float32)
    except:
        continue
    # obtain embedding vector for image
    embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

def distance(emb1, emb2):
    return np.sum(np.square(emb1 - emb2))

def show_pair(idx1, idx2):
    plt.figure(figsize=(8,3))
    plt.suptitle(f'Distance = {distance(embedded[idx1], embedded[idx2]):.2f}')
    plt.subplot(121)
    plt.imshow(load_image(metadata[idx1].image_path()))
    plt.subplot(122)
    plt.imshow(load_image(metadata[idx2].image_path()));

show_pair(2, 3)
show_pair(2, 12)

from sklearn.metrics import f1_score, accuracy_score

distances = [] # squared L2 distance between pairs
identical = [] # 1 if same identity, 0 otherwise

num = len(metadata)

for i in range(num - 1):
    for j in range(1, num):
        distances.append(distance(embedded[i], embedded[j]))
        identical.append(1 if metadata[i].name == metadata[j].name else 0)

distances = np.array(distances)
identical = np.array(identical)

thresholds = np.arange(0.03, 100.0, 0.01)

f1_scores = [f1_score(identical, distances < t) for t in thresholds]
acc_scores = [accuracy_score(identical, distances < t) for t in thresholds]

opt_idx = np.argmax(f1_scores)
# Threshold at maximal F1 score
opt_tau = thresholds[opt_idx]
# Accuracy at maximal F1 score
opt_acc = accuracy_score(identical, distances < opt_tau)

# Plot F1 score and accuracy as function of distance threshold
plt.plot(thresholds, f1_scores, label='F1 score');
plt.plot(thresholds, acc_scores, label='Accuracy');
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title(f'Accuracy at threshold {opt_tau:.2f} = {opt_acc:.3f}');
plt.xlabel('Distance threshold')
plt.legend();

dist_pos = distances[identical == 1]
dist_neg = distances[identical == 0]

plt.figure(figsize=(12,4))

plt.subplot(121)
plt.hist(dist_pos)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (pos. pairs)')
plt.legend();

plt.subplot(122)
plt.hist(dist_neg)
plt.axvline(x=opt_tau, linestyle='--', lw=1, c='lightgrey', label='Threshold')
plt.title('Distances (neg. pairs)')
plt.legend();



from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import LinearSVC
#
targets = np.array([m.name for m in metadata])
#
encoder = LabelEncoder()
encoder.fit(targets)
#
# # Numerical encoding of identities
y = encoder.transform(targets)
#
train_idx = np.arange(metadata.shape[0])
test_idx = np.arange(metadata.shape[0])

X_train = embedded[train_idx]
X_test = embedded[test_idx]

y_train = y[train_idx]
y_test = y[test_idx]

knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
# svc = LinearSVC()

knn.fit(X_train, y_train)
# svc.fit(X_train, y_train)
pickle.dump(knn, open('knn.sav', 'wb'))
# knn = pickle.load(open('knn.sav', 'rb'))
acc_knn = accuracy_score(y_test, knn.predict(X_test))
#acc_svc = accuracy_score(y_test, svc.predict(X_test))

# print(f'KNN accuracy = {acc_knn}, SVM accuracy = {acc_svc}')
# print(f'KNN accuracy = {acc_knn}')
# exit(0)
# knn = pickle.load(open('knn.sav', 'rb'))


key = cv2.waitKey(1) & 0xFF

import cv2
video_capture = cv2.VideoCapture(0)
# Check success
if not video_capture.isOpened():
    raise Exception("Could not open video device")
# Read picture. ret === True on success
font = cv2.FONT_HERSHEY_SIMPLEX
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
ret, frame = video_capture.read()
while ret == True:
    ret, frame = video_capture.read()
    cv2.imwrite("test/Random/image.png",frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    if len(faces)<1:
        print("No face")
        continue
    else:
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        plt.imshow(roi_color)
    metadata = load_metadata('test')
    embedded = np.zeros((metadata.shape[0], 128))

    for i, m in enumerate(metadata):
        img = load_image(m.image_path())
        img = align_image(img)
        # scale RGB values to interval [0,1]
        try:
            img = (img / 255.).astype(np.float32)
        except:
            continue
        # obtain embedding vector for image
        embedded[i] = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))[0]

    example_image = load_image(metadata[0].image_path())
    example_prediction = knn.predict([embedded[0]])
    example_identity = encoder.inverse_transform(example_prediction)[0]
    print(example_identity)
    cv2.putText(frame, str(example_identity), (230, 50), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imshow("image",frame)
    cv2.waitKey(2)
    cv2.destroyAllWindows()
    #time.sleep(2)
    plt.title(f'Recognized as {example_identity}');
    if key == ord("q"):
        cv2.destroyAllWindows()
        cap.release()
        break
# Close device
video_capture.release()
