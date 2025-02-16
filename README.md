# face_vectors
The goal of this Python service to detect faces and their time sequences in a video clip, trracking the same face throughout
the clip which is uploaded.  This is done by comparing their vectors using ChromaDB.  There is a dockerfile which should build this 
properly, and when run will expose an endpoint on port 5000.   The extracted face vectors are stored in chromaDB, which lets us query 
for other "close" faces frame-by-frame.   Simple debugging output displays the detected frames and which faceIDs are detected in 
each.
