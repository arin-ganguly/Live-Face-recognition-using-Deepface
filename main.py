##----------This is the Final Recognizer Code-------------##

import cv2
from deepface import DeepFace
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ---- CONFIG ---- #
embedded_file = r"E:\Fortuna_Video_Analytics_OnFace_&_FAISS\Embedded files\embeddings.pkl"
cosine_threshold = 0.3  # Lower for stricter match

# ---- Load embeddings ---- #
with open(embedded_file, "rb") as f:
    data = pickle.load(f)
known_embeddings = np.array(data["embeddings"])
known_labels = data["labels"]

# ---- Webcam ---- #
cap = cv2.VideoCapture(0)  # 0 for default camera

print("[INFO] Starting webcam face recognition... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    try:
        # Step 1: Face Detection 
        faces = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="mtcnn",
            enforce_detection=False,
            align=True
        )
    except Exception as e:
        print(f"[Detection Error] {e}")
        continue

    for face in faces:
        try:
            # Step 2: Get embedding using ArcFace
            embedding = DeepFace.represent(
                img_path=face["face"],
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False
            )[0]["embedding"]

            embedding = np.array(embedding).reshape(1, -1)
            similarities = cosine_similarity(embedding, known_embeddings)[0]

            best_idx = np.argmax(similarities)
            best_score = similarities[best_idx]
            confidence = best_score * 100

            if best_score >= cosine_threshold:
                matched_name = known_labels[best_idx]
                color = (0, 255, 0)
            else:
                matched_name = "Unknown"
                color = (0, 0, 255)

            # Step 3: Draw box + label
            bbox = face["facial_area"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            label = f"{matched_name} ({confidence:.2f}%)"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Step 4: Terminal output
            print(f"[MATCH] {matched_name} - Confidence: {confidence:.2f}%")
            # print(f"[MATCH] {matched_name} - cosine_threshold: {cosine_threshold}")

        except Exception as e:
            print(f"[Recognition Error] {e}")
            continue

    cv2.imshow("Webcam Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
