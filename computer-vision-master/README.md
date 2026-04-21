# Computer Vision Module

A hands-on module covering computer vision from image fundamentals to face authentication, built with OpenCV, YOLO, CLIP, and ChromaDB.

The module is organized into three areas. Each area can be studied independently, though the progression from one to the next is intentional.

## OpenCV: Images and Video

- [opencv/opencv_fundamentals.ipynb](opencv/opencv_fundamentals.ipynb): loading and manipulating images; BGR vs RGB, `uint8`, resizing, rotation, cropping, drawing, and saving.
- [opencv/opencv_image_processing.ipynb](opencv/opencv_image_processing.ipynb): classical image processing; filtering, convolution, edge detection, thresholding, morphology, contours, and CLAHE.
- [opencv/opencv_video.ipynb](opencv/opencv_video.ipynb): video processing from files and webcams; frame loops, codecs, background subtraction, and optical flow.

## YOLO: Object Detection

- [yolo/object_detection_and_yolo.ipynb](yolo/object_detection_and_yolo.ipynb): detection concepts and inference; bounding boxes, IoU, NMS, metrics, pre-trained inference, segmentation, speed comparison, and export.
- [yolo/yolo_custom_training.ipynb](yolo/yolo_custom_training.ipynb): fine-tuning YOLO; annotation format, dataset validation, training configuration, plots, transfer learning, and catastrophic forgetting.

## Embeddings, Databases, and Recognition

- [vectors_and_embeddings.ipynb](vectors_and_embeddings.ipynb): image embeddings; vector arithmetic, cosine vs Euclidean distance, CLIP, PCA, and links to NLP and RAG.
- [chromadb_intro.ipynb](chromadb_intro.ipynb): vector databases; HNSW search, cosine distance, metadata filtering, and ephemeral vs persistent clients.
- [face_recognition_pipeline.ipynb](face_recognition_pipeline.ipynb): end-to-end face verification; capture, detect, embed, store, verify, webcam workflow, and a similarity heatmap.

## Suggested Project: Face Authentication with a Webcam

The notebooks above provide all the pieces needed to build a small face authentication system — the kind used for attendance checking or room access.

**Goal**: enroll a set of people by capturing their face from a webcam, then verify their identity in real time.

**Suggested steps**:

1. Use the webcam pattern in [face_recognition_pipeline.ipynb](face_recognition_pipeline.ipynb) to capture several frames per person at enrollment.
2. For each frame, detect and crop the face using the YOLO pipeline shown in the same notebook.
3. Embed each crop with CLIP and store in a [ChromaDB persistent collection](chromadb_intro.ipynb).
4. At verification, capture a new frame, embed it, and query the database. Apply the 0.85 similarity threshold as a starting point — adjust based on your hardware and lighting.
5. Add a quality check (blur detection, brightness bounds) before embedding to avoid storing bad captures; the webcam section of [face_recognition_pipeline.ipynb](face_recognition_pipeline.ipynb) shows this pattern.

**Extensions to consider**:

- Use `yolov8n-face.pt` instead of the general model for tighter face crops.
- Enroll multiple frames per person and compare the query against all of them (soft voting).
- Store enrollment data with timestamps; flag embeddings older than N days as stale.
- Add a logging step for rejected attempts.

The gap between this notebook-based prototype and a real deployment is primarily reliability engineering, not algorithmic complexity, which makes it a useful project scope for learning.

## Repository Structure

```text
.
├── opencv/
│   ├── opencv_fundamentals.ipynb
│   ├── opencv_image_processing.ipynb
│   └── opencv_video.ipynb
├── yolo/
│   ├── object_detection_and_yolo.ipynb
│   └── yolo_custom_training.ipynb
├── vectors_and_embeddings.ipynb
├── chromadb_intro.ipynb
├── face_recognition_pipeline.ipynb
├── resources/
│   ├── images/           # static images used by the notebooks
│   │   ├── lenna.png
│   │   ├── baboon.png
│   │   ├── peppers.jpg
│   │   ├── bus.jpg
│   │   ├── scene1.jpg
│   │   ├── scene2.jpg
│   │   └── scene3.jpg
│   └── models/           # gitignored; downloaded YOLO weights
└── artifacts/
    ├── outputs/          # gitignored; generated images, videos, and YOLO runs
    └── face_db/          # gitignored; persistent ChromaDB demo data
```

## Environment

Python ≥ 3.13, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # install all dependencies from pyproject.toml
```

Key dependencies: `ultralytics`, `opencv-python`, `sentence-transformers`, `chromadb`, `scikit-learn`, `matplotlib`.
