from django.shortcuts import render
import json
import pymongo
import cv2
from deepface import DeepFace
import numpy as np
from base64 import b64decode
from django.http import JsonResponse
import base64
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["video_stream"]
collection = db["Person"]

known_faces = {}
for doc in collection.find():
    name = doc["firstname"]
    emb = doc["embedding"]
    known_faces[name] = emb

def index(request):
    return render(request,"index.html",{})

def process_video_frame(request):
    if request.method == "POST":
        recognised_faces = []
        
        # Read the raw binary data from the request body
        frame_bytes = request.body
        decoded_data = base64.b64decode(frame_bytes)
        # Convert the raw binary data into a NumPy array
        nparr = np.frombuffer(decoded_data, dtype=np.uint8)
    
        # Decode the NumPy array as an image using OpenCV
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        print(frame)
        # Perform face detection and recognition
        faces = cv2.CascadeClassifier('haarcascade_frontalface_default.xml').detectMultiScale(frame, 1.3, 5)
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y+h, x:x+w]
            
            # Represent the face using the Facenet model
            embedding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)
            dic = embedding[0]
            embedding = dic['embedding']
            embedding = np.array(embedding)

            if face.size == 0:
                continue

            # Compare the face embedding with known embeddings
            for name, known_embedding in known_faces.items():
                similarity_score = np.dot(embedding, known_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(known_embedding))
                if similarity_score > 0.7:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    recognised_faces.append({"name": name})
        
        # Prepare the response data
        response_data = {
            "processed_frame": frame,  # Convert NumPy array to list for JSON serialization
            "face_details": recognised_faces
        }
                   
        return JsonResponse(response_data)


def add_face(request):
    if request.method=="POST":
        image_base64 = doc["image"]
        nparr = np.frombuffer(image_base64, dtype=np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
        embedding = DeepFace.represent(img_np, model_name='Facenet')
        dic = embedding[0]
        emb = dic['embedding']
        emp = np.array(emb)