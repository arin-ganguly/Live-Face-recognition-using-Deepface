# Live Face Recognition using DeepFace

This project demonstrates a **real-time face recognition system** using the [DeepFace](https://github.com/serengil/deepface) library with live webcam input. It captures frames from a video stream, detects faces using **MTCNN**, generates face embeddings using **ArcFace**, and compares them to pre-saved embeddings to recognize individuals.

> ⚠️ **Note:** This implementation currently suffers from **low FPS (frames per second)**, especially due to the high computation time of MTCNN and ArcFace on each frame in real time without GPU acceleration.

---

## 📸 Demo

https://github.com/user-attachments/assets/240e8a4c-cc2a-4627-99d4-a7336a085a3d


---

## 🚀 Features

- Real-time face detection and recognition from webcam/IP camera
- Embedding generation using **ArcFace**
- Face detection using **MTCNN**
- Known vs Unknown classification using cosine similarity
- Modular structure for better scalability
- Custom embedding database (`embeddings.pkl`) for known individuals

---

## 🧠 Technologies Used

| Component        | Technology         |
|------------------|--------------------|
| Face Detection   | MTCNN (via DeepFace) |
| Face Recognition | ArcFace (via DeepFace) |
| Framework        | Python 3.11        |
| Face Matching    | Cosine Similarity  |
| Embedding Storage| Pickle (.pkl)      |
| Camera Input     | OpenCV             |
| Visualization    | OpenCV GUI         |

**Libraries:**
- `deepface`
- `opencv-python`
- `numpy`
- `pickle`
- `os`, `cv2`, `glob`, `datetime`

---

**Contributors:**

Arin Ganguly(https://github.com/arin-ganguly)

Ankan Das(https://github.com/Ankandas2004)

Abhijit Dey(https://github.com/IamAbhijit2004)

