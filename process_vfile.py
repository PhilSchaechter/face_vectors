
from flask import Flask, request, render_template, send_file
import subprocess
import os
import hashlib
import cv2
import numpy as np
import dlib
import chromadb
from chromadb.config import Settings

app = Flask(__name__)

# Define settings with the persist directory
settings = Settings(persist_directory='/storage/chroma_data')
chroma_slim = chromadb.Client(settings)
slim_collection = chroma_slim.create_collection(name="face_encodings")

clipid = 0

FRAMES_EXTRACT_PER_SEC = os.environ.get('FRAMES_EXTRACT_PER_SEC', 2)
SINGLE_CLIP_DETECT_THRESH = os.environ.get('SINGLE_CLIP_DETECT_THRESH', .1)
ANTI_STUTTER_SECONDS = os.environ.get('ANTI_STUTTER_SECONDS', 5)
ANTI_STUTTER_FRAMES = FRAMES_EXTRACT_PER_SEC * ANTI_STUTTER_SECONDSS

# Load the face detector and the face recognition model
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face_recognizer = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

@app.route('/return_frame', methods=['GET'])
def return_frame():
    frame = request.args.get('frame')
    path = f"/storage/{frame}"
    try:
        # Send the file with the proper MIME type for JPEG files
        return send_file(path, mimetype='image/jpeg')
    except FileNotFoundError:
    # If the file does not exist, return a 404 error
        return "File not found", 404


@app.route('/extract_frames', methods=['POST', 'GET'])
def extract_frames():
    if request.method != 'POST':
       return render_template('upload_video.html')

    print("Startup")
    global clipid
    clipid += 1
    personid = 1000;
    input_file = request.files['video']
    video_bytes = input_file.read()

    cid = "prefix_"

    # Extract the frames using FFmpeg
    output_file = cid + '_%04d.jpg'
    output = "<html><head><title>Processed Frames</title><head><body>"

    cmd = ['ffmpeg', '-i', '-', '-vf', 'fps={}'.format(FRAMES_EXTRACT_PER_SEC), '-q:v', '2', f"/storage/{output_file}"]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as process:
        # Pipe the input video file's content to FFmpeg via stdin
        try:
            process.stdin.write(video_bytes)
        finally:
            process.stdin.close()
            # Wait for FFmpeg to finish processing
            process.wait()

    fcount = 0
    for filename in os.listdir('/storage'):
        if filename.startswith(cid + '_'):
            fcount += 1
            frame = cv2.imread(f"/storage/{filename}")
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            #value = cv2.imencode('.jpg', frame)[1].tobytes()

            #frame = np.frombuffer(value, dtype=np.uint8)
            #frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

            # Convert the image to grayscale
            #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Convert directly to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces = detector(gray)

            if not faces:
               print("No faces in frame?")
               continue

            # Iterate through each detected face and draw a rectangle around it
            facecount=0
            for face in faces:
               x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
               cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

               # Get the face landmarks and compute the face encoding
               landmarks = predictor(gray, face)
               face_encoding = face_recognizer.compute_face_descriptor(frame, landmarks)
               face_encoding_list = [elem for elem in face_encoding]

               # query slimDB looking for this person
               results = slim_collection.query(
                       query_embeddings=face_encoding_list,
                       n_results=1
                       )
               if results['ids'][0]:
                       print(results)
                       closest_id = results['ids'][0][0]
                       distance = results['distances'][0][0]
                       frame_number = results['metadatas'][0][0]['frame_number']
                       if (distance < SINGLE_CLIP_DETECT_THRESH): 
                           person = results['metadatas'][0][0]['person']
                           print("found old person")
                       else:
                           print("made new person")
                           person = personid
                           personid += 1

                       print(f"On Frame: {fcount}\tID: {closest_id}\tFrom Frame\t{frame_number}\t same: {person}\tDistance: {distance}")
                      
               else:
                       print("made new person**")
                       person = personid
                       personid += 1
                       distance=0

               # insert into slimDB
               metadata=[] 
               metadata.append({"frame_number": fcount, "person": person})
               ids=f"{clipid}_{fcount}_{facecount}" 
               slim_collection.add(ids=ids, embeddings=face_encoding_list, metadatas=metadata)

               # Print the 128 vector face encoding
               #print("128 vector face encoding:", *face_encoding)
               #rect_faces = cv2.imencode('.jpg', frame)[1].tobytes()
               center_x = int((x1 + x2) / 2)
               center_y = int((y1 + y2) / 2)
               font = cv2.FONT_HERSHEY_SIMPLEX
               font_scale = 1
               text_color = (0, 0, 255)
               thickness=2
               cv2.putText(frame, str(person) +" "+ str(distance)[:4], (center_x, center_y), font, font_scale, text_color, thickness)



               facecount += 1

            print(f"Frame {fcount}: found {facecount}")
            cv2.imwrite(f"/storage/{filename}", frame)
            output += f"<hr><img src=\"/return_frame?frame={filename}\">"

    for i in range(1000, person):
        midpoint_person = []
        print(f"Post-processing for person {i}")
        results = slim_collection.get(
                    where={"person": i},
                    include=["documents", "metadatas", "embeddings"]
        )
        sorted_results = sorted(
            zip(results["ids"], results["documents"], results["metadatas"], results["embeddings"]),
            key=lambda x: x[2]["frame_number"]  # Sort by the metadata field 'frame_number'
        )


        for result in zip(sorted_results["ids"], sorted_results["documents"], sorted_results["metadatas"], sorted_results["embeddings"]):
            midpoint_person.append(sorted_results["embeddings"])

        midpoint_np = np.array(midpoint_person)
        if len(midpoint_np) > 0:
            midpoint = np.mean(midpoint_np, axis=0)
            # find frame closest to midpoint, and extract face
            # insert person in persistent dbs.
        else:
            print(f"No embeddings for person {i}?")


    # Do something with the extracted frames
    print("Received frames:", fcount)

    return output

if __name__ == '__main__':
    app.run(host="0.0.0.0")


