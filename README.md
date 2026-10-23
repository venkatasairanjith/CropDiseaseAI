# AI-Based Crop Disease Detection and Management System
**Course:** E1CSA313 Design Project - 1  
**Institution:** Alliance University  
**Team Members:**  
1. Sai Ranjith - 2411021060729  
2. A. Akhileswara achari - 2411021060747  
3. S. Haneesh Ranganath - 2411021060722  
4. D. Karunakar - 2411021060757  
**Mentor:** Dr. Devika P, Assistant Professor  

---

## 📌 Proposed Architecture Implementation (Parts 1 to 4)

This project implements the core computational intelligence and web interface as specified in **Page 11 of the Design Project Review Presentation**:

1. **Part 1: Crop Leaf Image Acquisition (`core/part1_acquisition.py`)**  
   - Ingests foliar images via device upload, camera, or PlantVillage benchmark samples.
2. **Part 2: Image Preprocessing (`core/part2_preprocessing.py`)**  
   - Standardizes resolution to $224 \times 224$ px.  
   - Applies OpenCV bilateral filtering for edge-preserving lesion denoising.  
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on the Lightness channel.  
   - Computes standard ImageNet normalization.
3. **Part 3: AI Disease Classification & Severity Estimation (`core/part3_classifier.py`)**  
   - Pretrained CNN (MobileNetV2 Transfer Learning) classifies host crop & disease.  
   - Quantifies foliar damage into calibrated **Severity Tiers** (*Mild, Moderate, Severe, or Healthy*).
4. **Part 4: Crop Health Assessment (`core/part4_assessment.py`)**  
   - Multimodal fusion: Evaluates visual disease severity alongside 4 continuous IoT environmental telemetry streams (*Soil Moisture %, Temperature °C, Humidity %, Soil pH*).  
   - Computes a consolidated **Crop Health Index (0 - 100)**.  
   - Prescribes actionable chemical formulations and precise dosages (e.g., *Chlorothalonil 75 WP @ 2.0 g/L*).

---

## 🚀 How to Run the Web Dashboard Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the Streamlit Web Application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`.

---

## ☁️ 1-Click Free Cloud Deployment (Streamlit Community Cloud)

1. Push this repository to your GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: CropDiseaseAI Parts 1 to 4 Web Dashboard"
   git branch -M main
   git remote add origin https://github.com/GaMeBoY-21/CropDiseaseAI.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io) and log in with your GitHub.
3. Click **"New app"** -> Select your repository `CropDiseaseAI` -> Main file `app.py`.
4. Click **Deploy**! In 60 seconds you will have a live public URL to present to the panel.
