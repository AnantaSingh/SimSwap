import pickle
import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import insightface
from insightface.app import FaceAnalysis
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained SVM model
model_path = "output/face_models_full.pkl"
with open(model_path, 'rb') as f:
    models = pickle.load(f)
svm = models['svm']
pca = models['pca']
scaler = models['scaler']
encoder = models['encoder']

# Function to extract face and predict
def predict_face(image_path, app, embedder):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load: {image_path}")
        return None
    faces = app.get(img)
    if len(faces) == 0:
        print(f"No face detected in: {image_path}")
        return None
    face = cv2.resize(img, (160, 160))  # Match FR input size
    face = face.astype('float32') / 255.0
    face_tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(torch.device('cpu'))
    with torch.no_grad():
        embedding = embedder(face_tensor).cpu().numpy()
    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = scaler.transform(embedding)
    embedding = pca.transform(embedding)
    prediction = svm.predict(embedding)
    return encoder.inverse_transform(prediction)[0]

# Initialize InsightFace for face detection in deepfakes
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(320, 320), det_thresh=0.3)
embedder = InceptionResnetV1(pretrained='vggface2', classify=False).eval().to(torch.device('cpu'))

# Directory for deepfake images
deepfake_dir = "output/deepfakes"

# Test deepfake images
results = []
for celeb_folder in os.listdir(deepfake_dir):
    celeb_folder_path = os.path.join(deepfake_dir, celeb_folder)
    if os.path.isdir(celeb_folder_path):
        for img_file in os.listdir(celeb_folder_path):
            img_path = os.path.join(celeb_folder_path, img_file)
            if os.path.isfile(img_path) and img_file.lower().endswith('.jpg'):
                true_label = celeb_folder  # Use folder name as true label
                predicted_label = predict_face(img_path, app, embedder)
                if predicted_label is not None:
                    results.append({
                        'image_path': img_path,
                        'true_label': true_label,
                        'predicted_label': predicted_label,
                        'correct': true_label == predicted_label
                    })
                    print(f"Image: {img_path} | True: {true_label} | Predicted: {predicted_label} | Correct: {true_label == predicted_label}")

# Calculate metrics
total_deepfakes = len(results)
correct_predictions = sum(1 for r in results if r['correct'])
misidentification_rate = 1 - (correct_predictions / total_deepfakes) if total_deepfakes > 0 else 0
print(f"Total deepfake images tested: {total_deepfakes}")
print(f"Correct predictions: {correct_predictions}")
print(f"Misidentification rate: {misidentification_rate:.2f}")

# Visualize results
output_vis_dir = "output"
os.makedirs(output_vis_dir, exist_ok=True)

# Confusion Matrix for deepfake misidentification
true_labels = [r['true_label'] for r in results]
pred_labels = [r['predicted_label'] for r in results]
cm = confusion_matrix(true_labels, pred_labels, labels=encoder.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=encoder.classes_, yticklabels=encoder.classes_, cmap='Blues')
plt.title('Deepfake Misidentification Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig(os.path.join(output_vis_dir, 'deepfake_confusion_matrix.png'))
plt.close()

# Bar plot of misidentification rate per class
misid_per_class = {}
for label in encoder.classes_:
    true_count = sum(1 for r in results if r['true_label'] == label)
    incorrect_count = sum(1 for r in results if r['true_label'] == label and not r['correct'])
    misid_per_class[label] = incorrect_count / true_count if true_count > 0 else 0
plt.figure(figsize=(10, 6))
plt.bar(misid_per_class.keys(), misid_per_class.values())
plt.title('Misidentification Rate per Class for Deepfakes')
plt.xlabel('Class')
plt.ylabel('Misidentification Rate')
plt.xticks(rotation=45)
plt.savefig(os.path.join(output_vis_dir, 'deepfake_misid_rate.png'))
plt.close()

# ROC-like curve (approximated using misclassification)
from sklearn.metrics import roc_curve, auc
y_test_bin = label_binarize([r['true_label'] for r in results], classes=encoder.classes_)
y_score = label_binarize([r['predicted_label'] for r in results], classes=encoder.classes_)
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(len(encoder.classes_)):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
plt.figure(figsize=(8, 6))
for i, label in enumerate(encoder.classes_):
    plt.plot(fpr[i], tpr[i], label=f'{label} (AUC = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curves for Deepfake Misidentification')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.savefig(os.path.join(output_vis_dir, 'deepfake_roc_curves.png'))
plt.close()