# 🚀 Smart Manufacturing Machines Efficiency Prediction

![Project Banner](./static/Project.jpg)  
*Empowering factories with intelligent predictions for enhanced productivity.*

---

## 📌 Project Overview

The **Smart Manufacturing Machines Efficiency Prediction** project is a robust, full-stack solution that blends **Machine Learning**, **Cloud Infrastructure**, and **DevOps automation** to **predict the efficiency level of manufacturing machines in real-time**.

This project was developed in a real cloud-based DevOps environment on **Google Cloud Platform (GCP)** using **Kubernetes**, **Jenkins**, and **ArgoCD** for a GitOps CI/CD strategy. Due to GCP’s free tier quota limit, the original GCP deployment has been decommissioned as of **June 19, 2025**, but the complete solution is still **live and functional on Render**.

👉 [Live Project Demo](https://smart-manufacturing-machines-efficiency.onrender.com/)

---

## ✨ Key Highlights

✅ **Real-time Machine Learning Predictions** — Predicts machine performance (Low, Medium, High) using live inputs.  
✅ **XGBoost-Powered Model** — Trained on synthetic industrial data with **92% accuracy**.  
✅ **Cloud-Native & Scalable** — Containerized using **Docker**, orchestrated via **Kubernetes**.  
✅ **GitOps & CI/CD** — Fully automated deployment pipeline using **ArgoCD** and **Jenkins**.  
✅ **Web Interface** — User-friendly Flask-based interface for interactive use.

---

## 🧠 Technologies & Tools

| Category        | Tools & Frameworks                                    |
|----------------|--------------------------------------------------------|
| Programming     | Python                                                |
| ML & Data       | XGBoost, Scikit-learn, Pandas, Joblib                 |
| Web Framework   | Flask                                                 |
| DevOps/CI-CD    | Docker, Jenkins, Kubernetes, ArgoCD                   |
| Cloud & Hosting | GCP (development), Render (production deployment)     |

---

## 🏗️ Project Architecture

```
Smart-Manufacturing-Machines-Efficiency-Prediction/
├── artifacts/
│   ├── model/
│   ├── processed/
│   └── raw/
├── config/
│   ├── pycache/
│   ├── init.py
│   └── paths_config.py
├── logs/
├── manifests/
│   ├── deployment.yaml
│   └── service.yaml
├── notebook/
│   └── test.ipynb
├── pipeline/
│   ├── training_pipeline.py
│   └── smart_manufacturing_pipeline.py
├── src/
│   ├── pycache/
│   ├── init.py
│   ├── custom_exception.py
│   ├── data_ingestion.py
│   ├── data_processing.py
│   ├── logger.py
│   └── model_training.py
├── static/
│   └── css/
│       └── styles.css
├── templates/
│   └── index.html
├── .env
├── .gitignore
├── app.py
├── Dockerfile
├── ETL-Pipeline.py
├── Jenkinsfile
├── LICENSE
├── README.md
├── requirements.txt
└── setup.py
```

---

## 🔧 Installation & Run Locally

```bash
# Clone this repository
git clone https://github.com/FaheemKhan0817/Smart-Manufacturing-Machines-Efficiency-Prediction.git
cd Smart-Manufacturing-Machines-Efficiency-Prediction

# Install Python dependencies
pip install -r requirements.txt

# Set environment variables if needed
touch .env

# Build and run the Docker container
docker build -t faheemkhan08/gitops-project:latest .
docker run -p 5000:5000 faheemkhan08/gitops-project:latest

# Open your browser at http://localhost:5000
```

---

## 🚀 Usage Guide

- **Input Features**:  
  `Vibration_Hz`, `Packet_Loss_Perc`, `Production_Speed_units_per_hr`  
- **Output**:  
  One of `Low`, `Medium`, or `High` efficiency levels predicted by the model.

💡 Try it live here → [https://smart-manufacturing-machines-efficiency.onrender.com/](https://smart-manufacturing-machines-efficiency.onrender.com/)

---

## 🛠️ Development Lifecycle

1. **Data Simulation**: Created synthetic dataset simulating factory machine stats.
2. **Data Preprocessing**: Cleaned and transformed using `scikit-learn`.
3. **Modeling**: Built and tuned an `XGBoost` classifier.
4. **Containerization**: Dockerized the app for consistency across environments.
5. **Deployment**:
    - **Kubernetes** for orchestration
    - **ArgoCD** for GitOps-driven deployment
    - **Jenkins** for automated CI/CD pipeline
6. **Hosting**: Switched from **GCP VM** to **Render** due to quota limits.

---

## 🧩 Challenges Faced

- **💸 GCP Quota Limitations**: Resolved by shifting deployment to Render.
- **🐳 Docker Path Issues**: Fixed `FileNotFoundError` by bundling correct directories into the image.
- **🔐 External Access**: Handled ingress using Minikube locally and Render for public access.

---

## 📜 License

This project is licensed under the [MIT](LICENSE/) 

---

## 🤝 Let's Connect!

I'm actively seeking opportunities in **Machine Learning**, **MLOps**, and **Cloud/Data Engineering**.

- 🔗 [LinkedIn](https://linkedin.com/in/faheemkhanml)
- 💻 [GitHub](https://github.com/FaheemKhan0817)
- 📧 Let’s build something impactful together!

---
