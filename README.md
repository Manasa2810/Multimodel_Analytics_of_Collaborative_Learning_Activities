# Multimodel Analytics of Collaborative Learning Activities  
**Indian Institute of Technology Gandhinagar**  
Under the supervision of **Prof. Aditi Kothiyal**

---

## Overview

This project presents a full **multimodal analytics pipeline** designed to study how students collaborate during hands-on learning tasks.  
Using synchronized **video + audio**, we extract behavioral signals from:

- Hand–object interactions  
- Gaze and attention  
- Facial emotions  
- Speech dynamics  

These signals are fused into per-person feature vectors and analyzed using **PCA** and **K-means clustering** to identify meaningful behavioral profiles such as *productive*, *effective*, and *passive* collaborators.

The pipeline outputs include:
- Annotated videos  
- JSON interaction logs  
- Per-person feature CSV  
- Gaze/emotion/speech features  
- Clustering visualizations  
- Behavioral summaries and insights  

---

##  Key Contributions

### **1. Complete Hand–Object Interaction Pipeline**

Built using:
- **MediaPipe Hands**  
- **HSV object detection**  
- Centroid tracking  
- Temporal smoothing and interaction logic  

Detects:
- Reach, Touch, Grasp, Hold  
- Pass / Shared-object interactions  
- Workspace region per person  
- Active hand assignment  
- Interaction frequency & duration patterns  

Outputs include:
- Annotated video  
- Event-level JSON log  
- Feature matrix for modeling behavior  

---

### **2. Multimodal Behavioral Feature Extraction**

| Modality | Extracted Features |
|---------|---------------------|
| **Hands** | grasp counts, reach/touch frequency, interaction durations, hand speed, workspace area |
| **Objects** | unique objects touched, switch rate, color preference entropy, pass/shared metrics |
| **Gaze** | fixation distribution, gaze heatmaps, joint attention |
| **Emotions** | emotion proportions, emotion transitions, stability |
| **Speech** | speaking time, pauses, overlap speech ratio |

Together, these form a high-resolution behavioral fingerprint for each participant.

---

## Clustering & Collaboration Profiles

After constructing the multimodal feature matrix:

1. Features were standardized  
2. Reduced using **PCA**  
3. Clustered using **K-means** (validated using elbow + silhouette)  
4. Clusters interpreted using centroid heatmaps + PCA scatter plots  

### **Final Clustering Result: 3 Distinct Collaboration Profiles**

Based on the PCA plots, cluster distributions, and feature gradients:

---

### **Cluster 1 — Productive Learners (High Engagement Group)**  
**Behavioral Signature:**
- Highest number of interactions (reach/touch/grasp)  
- Longer grasp durations → deeper object manipulation  
- High workspace coverage  
- Strong hand speed + movement variety  
- More object-switching and active exploration  
- High gaze transitions (shifting attention frequently)  
- Moderate speech contribution (balanced talking + working)

**Interpretation:**  
These students engaged deeply with the task materials, explored objects actively, and demonstrated high behavioral involvement.

---

### **Cluster 2 — Effective Collaborators (Balanced Group)**  
**Behavioral Signature:**
- Moderate number of interactions  
- Smooth interaction rhythm (consistent gaps)  
- Balanced speaking–listening patterns  
- Higher joint attention alignment (from gaze)  
- Lower unnecessary object switching  
- Stable emotional patterns (rare spikes)  

**Interpretation:**  
These students worked efficiently, coordinated well with the partner, and displayed steady and structured collaboration patterns.

---

### **Cluster 3 — Passive Participants (Low Engagement Group)**  
**Behavioral Signature:**
- Few reach/touch/grasp events  
- Lowest workspace coverage  
- Minimal object exploration  
- Limited gaze shifts (narrow attention)  
- Long pauses or low speech activity  
- Short or no grasp durations  
- Very low switching across objects  

**Interpretation:**  
These students contributed minimally to the task, interacted infrequently, and often played an observing role.

---

## What PCA & K-means Revealed

From the notebook and final PCA scatter plots:

- **PC1** strongly captured *interaction intensity*  
  (grasp counts, durations, workspace coverage, hand speed).  

- **PC2** captured *attention + speech dynamics*  
  (fixation spread, emotion variability, speech overlap).

Clusters were cleanly separable along PC1 and PC2, showing that:
- **Hands + objects → explain engagement**  
- **Gaze + speech → explain collaboration quality**  

This validates the strength of multimodal fusion.

---

## Feature Highlights Used in Clustering

### **Interaction Metrics**
- Total interactions  
- Grasp/Touch/Reach counts  
- Interaction frequency (per minute)  
- Interaction gaps & response times  

### **Movement & Workspace**
- Average hand speed  
- Speed variability  
- Workspace convex hull  
- Workspace overlap ratio  

### **Object Behavior**
- Color preference entropy  
- Unique object touches  
- Object switch rate  
- Shared-object frames  

### **Gaze & Emotion**
- Fixation density  
- Gaze variability  
- Emotion distributions  

### **Speech**
- Speaking duration  
- Pause statistics  
- Speech overlap %  

---

##  Results Summary (Final Evaluation)

- **Multimodal features consistently produced tighter, well-separated clusters** compared to using any single modality.  
- Interaction features such as **grasp duration**, **workspace area**, and **object-switch rate** had the strongest cluster influence.  
- Gaze and speech added nuance, separating *productive* from *effective* collaborators.  
- Passive participants showed low activation across all modalities — forming a clean, distinct cluster.  

These findings strongly support the value of multimodal analytics for understanding collaborative learning behavior.

---

## Challenges

- Hand occlusions and rapid movements created landmark noise  
- Gaze estimation struggled with low-resolution faces  
- Audio noise caused imperfect speech segmentation  
- HSV ranges required fine-tuning for each environment  
- Emotions difficult to detect from small facial regions  

Despite these, the final system produced consistent and interpretable behavioral profiles.



---
