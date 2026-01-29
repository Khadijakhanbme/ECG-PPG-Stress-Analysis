# Stress Detection from ECG & Finger PPG (MAUS Dataset)

This repository contains a complete **signal-processing and machine learning pipeline** for **stress detection** using **electrocardiogram (ECG)** and **finger photoplethysmogram (PPG)** signals from the **MAUS (Multimodal Assessment of User Stress)** dataset.

The project was developed as part of an **Assistive Devices** coursework and follows a design philosophy suitable for **embedded and microcontroller-oriented** feature extraction and inference.

---

## Project Highlights

- **Modalities:** ECG + finger PPG  
- **Segmentation:** Fixed **10-second windows** (deployment-friendly)  
- **Pipeline:**  
  preprocessing → segmentation → feature extraction → feature analysis → model training → evaluation  
- **Goal:** Classify **stress vs non-stress** states using **engineered physiological features**

---

## Purpose of the Study

The primary objectives of this project were to:

- Design a **physiologically interpretable** signal processing pipeline for ECG and PPG  
- Extract **computationally efficient features** suitable for **real-time and embedded systems**  
- Investigate **stress vs non-stress classification** using a **reduced cognitive workload paradigm**  
- Evaluate **classical machine learning models** under **resource-aware constraints**  
- Compare obtained results with existing **ECG–PPG-based stress detection literature**

---

## Stress Paradigm and Experimental Rationale

The analysis focuses exclusively on **two cognitive workload conditions**:

- **0-back:** Low cognitive load (**non-stress condition**)  
- **2-back:** High cognitive load (**stress condition**)  

The **1-back condition** was intentionally excluded.

### Rationale for Binary Selection

- Embedded prototype requirements target only **clear stress / non-stress states**
- Enables **robust class separation** under **real-time constraints**
- Simplifies **decision logic** for **microcontroller-based deployment**
- Aligns with **practical assistive-device scenarios**, rather than multi-class cognitive modeling

---

## Dataset

- **Dataset:** MAUS (Multimodal Assessment of User Stress)  - Open Source
- **Signals Used:**  
  - Electrocardiogram (**ECG**)  
  - Photoplethysmogram (**PPG**)  
- **Labels Used:**  
  - Cognitive workload labels corresponding to **0-back** and **2-back** tasks only.
  - 
---

## Methodology

### Signal Preprocessing
- Noise reduction and filtering applied **independently** to ECG and PPG
- Signal morphology preserved to retain **autonomic and cardiovascular dynamics**

### Segmentation
- **Fixed-length window segmentation**
- 10 sec window size selected to match **real-time acquisition** and **embedded processing constraints**

### Feature Extraction
- **Time-domain** and **morphology-based** features extracted from ECG and PPG
- Feature design prioritizes:
  - Physiological interpretability
  - Low computational complexity
  - Suitability for **microcontroller implementation**

### Feature Analysis and Reduction
- **Correlation analysis** to study feature–label relationships
- **Principal Component Analysis (PCA)** used to:
  - Analyze feature redundancy
  - Reduce dimensionality while preserving variance
- **Outlier detection** performed in reduced feature space to improve robustness

### Machine Learning Models
The following **classical ML models** were evaluated for **binary stress classification**:

- Logistic Regression  
- Random Forest  
- XGBoost  

---

## Results

### Classification Performance
- **Random Forest** achieved the best overall performance
- Accuracy and confusion matrix analysis demonstrate:
  - Clear separation between **low** and **high cognitive load** states
  - Competitive performance of 76% relative to reported **ECG–PPG literature**

### Key Observations
- Feature-based multimodal ECG–PPG approaches remain effective **without deep learning**
- Combining ECG and PPG features improves robustness over single-modality analysis
- The pipeline is suitable for **real-time embedded assistive-device applications**

---

## Conclusion

This project demonstrates a **multimodal ECG–PPG stress detection framework** integrating:

- Physiological signal processing  
- Feature-based machine learning  
- Embedded-system design considerations  

While full **on-device inference** was limited by hardware constraints, **real-time signal acquisition and feature extraction** were successfully achieved, validating the feasibility of deployment in **wearable or assistive health technologies understanding.

The results support the use of **lightweight, interpretable models** for stress monitoring in **resource-constrained environments**.

---

Embedded firmware and microcontroller-specific implementations are maintained in a **separate repository** to ensure **clarity and modularity**.
