
#import OpenCV module
import cv2
#import os module for reading training data directories and paths
import os
#import numpy to convert python lists to numpy arrays as
#it is needed by OpenCV face recognizers
import numpy as np
import random

#there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = ["", "Bolsomito", "Seu Madruga", "Idris Elba"]

#function to detect face using OpenCV
def detect_face(img):
    #convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    haar_face_cascade = cv2.CascadeClassifier('opencv-files/haarcascade_frontalface_alt.xml')

    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5);

    #if no faces are detected then return original img
    if len(faces) == 0:
        return None, None


    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]



def prepare_training_data(data_folder_path, quant_training_images, arrayTest):

    dirs = os.listdir(data_folder_path)

    #list to hold all subject faces
    faces = []
    #list to hold labels for all subjects
    labels = []

    #let's go through each directory and read images within it
    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))


        subject_dir_path = data_folder_path + "/" + dir_name

        #get the images names that are inside the given subject directory
        subject_images_names = os.listdir(subject_dir_path)
        subject_images_names = list(set(subject_images_names).difference(arrayTest))
        subject_images_names = sorted(subject_images_names)
        print (subject_images_names)
        c = 0
        for image_name in subject_images_names:
            #ignore system files like .DS_Store
            if image_name.startswith("."):
                continue

            #build image path
            #sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            #read image
            image = cv2.imread(image_path)


            #detect face
            face, rect = detect_face(image)

            if face is not None:
                face = cv2.resize(face, (400, 500))
                #add face to list of faces
                faces.append(face)
                #add label for this face
                labels.append(label)
            else :
                print (image_path)
            c += 1
            if c == quant_training_images: break

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels


"""print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")

#print total faces and labels
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
"""

face_recognizers = {
    "LBPH": cv2.face.LBPHFaceRecognizer_create(),
    "EigenFace": cv2.face.EigenFaceRecognizer_create(),
    "FisherFace":  cv2.face.FisherFaceRecognizer_create()
    }


def draw_rectangle(img, rect):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 4)

#function to draw text on give image starting from
#passed (x, y) coordinates.
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_PLAIN, (x+y)/100.0, (255, 0, 255), 5)


def predict(test_img, person_name, face_recognizer):
    #make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    #detect face from the image
    face, rect = detect_face(img)
    face = cv2.resize(face, (400, 500))

    #predict the image using our face recognizer
    #confiança é distância = quanto menor, melhor -> houve maior similaridade
    # cv2.imshow("gg", cv2.resize(face, (400, 500)))
    # cv2.waitKey(0)

    label, confidence = face_recognizer.predict(face)

    #print subjects[label]
    label_text = "unknown"
    #if confidence <= 70 and subjects[label] == person_name:
    if subjects[label] == person_name:
        #get name of respective label returned by face recognizer
        label_text = subjects[label]
        correct_predictions[label_text] += 1

    #draw a rectangle around face detected
    draw_rectangle(img, rect)
    #draw name of predicted person
    draw_text(img, label_text, rect[0], rect[1]-5)

    return img

print("Predicting images...")

def accuracy(person_name, arrayTest, face_recognizer, correct_predictions):

    #load test images
    for image_name in arrayTest:
        directory = ""

        if (person_name == "Bolsomito"):
            directory = "s1"
        elif (person_name == "Seu Madruga"):
            directory = "s2"
        else:
            directory = "s3"
        #print "%s:imagem %d" % (person_name, i)

        img_file = "training-data/" + directory + "/" + image_name

        test_img = cv2.imread(img_file)
        #cv2.imshow("gg", cv2.resize(test_img, (400, 500)))
        #cv2.waitKey(0)


        #perform a prediction
        predicted_img = predict(test_img, person_name, face_recognizer)
    cv2.destroyAllWindows()
    print (correct_predictions)
    print (person_name)
    accuracy = correct_predictions[person_name] / float(len(arrayTest))
    print ("accuracy for %s is: %.2f" % (person_name, accuracy * 100))

def defineTestImagesArray(numberOfTestImages):
    #Set the seed for the experiment
    random.seed(1)
    numberOfImages = 30
    arrayTest = []

    i = 0
    #Be careful here, this could generate a runtime
    while i != numberOfTestImages:
        value = random.randint(1, numberOfImages + 1)
        value = str(value) + '.jpg'
        if (value not in arrayTest):
            arrayTest.append(value)
            i += 1
    return arrayTest

training_sample_levels = [5,20]

for face_recognizer in face_recognizers:
    for quant_training_images in training_sample_levels:
        print("Preparing data...")
        arrayTest = defineTestImagesArray(10)
        print (arrayTest)
        faces, labels = prepare_training_data("training-data", quant_training_images, arrayTest)
        print("Data prepared")

        # print total faces and labels
        print("Total faces: ", len(faces))
        print("Total labels: ", len(labels))

        recognizer = face_recognizers[face_recognizer]
        recognizer.train(faces, np.array(labels))
        for subject in subjects:
            correct_predictions = {"Bolsomito": 0, "Seu Madruga": 0, "Idris Elba": 0}
            if len(subject) != 0:
                print ("Levels being used: ")
                print ("Recognizer: %s" % face_recognizer)
                print ("Quantity of training images: %d" % quant_training_images)
                print ("Subject: %s" % subject)

                accuracy(subject, arrayTest, recognizer, correct_predictions)

print("Prediction complete")


img_file = "test-data/%s-test/test%d.jpg" % ("Seu Madruga", 1)
test_img = cv2.imread(img_file)

#perform a prediction
"""predicted_img = predict(test_img, "Seu Madruga")

#display images
cv2.imshow(subjects[2], cv2.resize(predicted_img, (predicted_img.shape[1], predicted_img.shape[0])))
cv2.waitKey(0)
cv2.destroyAllWindows()
"""
