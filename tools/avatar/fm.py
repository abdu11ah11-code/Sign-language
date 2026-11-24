"""fm.py IMAGE OUT.json — MediaPipe Face Mesh landmarks (pixels) for one image."""
import sys, json, cv2, mediapipe as mp
img = cv2.imread(sys.argv[1]); h, w = img.shape[:2]
with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, refine_landmarks=False, max_num_faces=1,
                                     min_detection_confidence=0.2) as fm:
    r = fm.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
if not r.multi_face_landmarks:
    print("NO FACE"); sys.exit(1)
pts = [[p.x * w, p.y * h, p.z * w] for p in r.multi_face_landmarks[0].landmark]
json.dump({"size": [w, h], "pts": pts}, open(sys.argv[2], "w")); print("FOUND", len(pts))
