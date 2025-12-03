# Multimodel Analytics of Collaborative Learning Activities  
**Indian Institute of Technology Gandhinagar**  
Under the supervision of **Prof. Aditi Kothiyal**

---

## 🧩 Overview

This project builds a complete **multimodal analytics pipeline** to study collaborative learning behavior using synchronized **video and audio** recordings.  
We extract, combine, and analyze multiple behavioral signals — including **hand–object interactions**, **eye gaze**, **facial emotions**, and **speech activity** — to construct rich per-person feature representations.

These features are then reduced using **PCA** and clustered using **K-means** to reveal meaningful collaboration patterns such as *productive*, *effective*, and *passive* learners.

The pipeline generates:
- Annotated videos  
- JSON interaction logs  
- Per-person feature matrices  
- Gaze, speech, and emotion features  
- Clustering visualizations  
- Behavioral summaries  

---

## ⭐ Key Contributions

### **1. Robust Hand–Object Interaction Pipeline**

Designed and implemented a stable pipeline that uses:
- **MediaPipe Hands**  
- **HSV-based object detection**  
- Centroid tracking  
- Temporal smoothing and rule-based logic  

It detects:
- Reach  
- Touch  
- Grasp  
- Hold  
- Object pass/shared usage  
- Active hand(s) per person  
- Workspace coverage and overlap  

The pipeline outputs:
- An annotated video highlighting interactions  
- A detailed frame-level JSON event log  
- A structured feature CSV for behavior modeling  

---

### **2. Multimodal Behavioral Feature Extraction**

| Modality | Features Extracted |
|---------|---------------------|
| **Hands** | interaction counts, grasp durations, hand speed, movement variability, workspace area |
| **Objects** | object switch rate, unique objects touched, color entropy, pass/shared-object metrics |
| **Gaze** | fixation regions, gaze heatmaps, visual attention distribution, joint attention |
| **Emotion** | emotion proportions, transitions, stability indicators |
| **Speech** | speaking time, pause lengths, speech overlap, speaking–listening balance |

All modality-specific features are fused to create a comprehensive behavior profile for each participant.

---

## 🧠 Clustering & Behavioral Profiling

Using the aggregated feature matrix:
1. Features are standardized  
2. PCA is applied to reduce dimensionality  
3. K-means clustering is performed using silhouette and elbow validation  
4. Cluster visualizations are generated  

The clusters reveal distinct profiles such as:
- **Productive learners** – high engagement, frequent interaction, strong coordination  
- **Effective collaborators** – balanced turn-taking, steady rhythmic engagement  
- **Passive participants** – minimal interaction, weak gaze/speech transitions  

All results, figures, and cluster explanations are included in the notebook:  
**`Kmeans_clustering.ipynb`**

---

## 📊 Feature Highlights

Every participant receives a comprehensive feature vector that includes:

### **Interaction Metrics**
- Total number of interactions  
- Reach, touch, grasp counts  
- Average and total grasp duration  
- Interaction frequency (per minute)  
- Interaction gaps  
- Response times  

### **Movement & Workspace**
- Mean and variance of hand speed  
- Convex-hull workspace area  
- Workspace overlap ratio between partners  

### **Object Behavior**
- Object switch rate  
- Number of unique objects interacted with  
- Color preference entropy  
- Object pass and shared-object counts  

### **Gaze & Emotion (optional modules)**
- Fixation heatmaps  
- Region-wise attention  
- Emotion distributions and transitions  

### **Speech Features**
- Total speaking duration  
- Pause-length distributions  
- Overlap speech ratio  

---

## 📈 Results Summary

Key findings from the final analysis:

- **Multimodal fusion significantly improves clustering clarity**, compared to using a single modality.  
- Interaction-based features (grasp durations, workspace coverage, etc.) were highly discriminative between clusters.  
- Gaze distribution and speech dynamics strongly supported identifying effective vs. passive participants.  
- PCA scatterplots and feature heatmaps show well-separated clusters aligned with intuitive collaboration behaviors.  

All detailed figures and interpretations are available in the poster and notebook.

---

## ⚠️ Challenges & Learnings

- Hand detection becomes unstable with occlusions or rapid movement  
- Gaze estimation is sensitive to video resolution and face angle  
- Audio noise affects speech segmentation reliability  
- HSV object detection requires lighting-dependent tuning  
- Emotion recognition accuracy drops for small or blurred faces  

Despite these challenges, the combined pipeline produces robust and interpretable multimodal behavioral insights.

---


