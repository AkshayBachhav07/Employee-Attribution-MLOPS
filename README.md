# Employee Attrition Prediction System (End-to-End MLOps)

An end-to-end Machine Learning Operations (MLOps) microservice that predicts employee attrition risk based on compensation, job satisfaction, work-life balance, organizational tenure, and working hours. Built with Scikit-learn, served via FastAPI, containerized using Docker, validated with Pytest, and automated through GitHub Actions CI/CD.

---

## 📌 Repository Architecture

```text
Employee-Attribution-MLOPS/
├── .github/
│   └── workflows/
│       └── ci-cd.yml          # Automated CI/CD (lint, train, test, Docker build)
├── data/
│   └── dataset.csv            # Tabular employee dataset
├── models/
│   └── model.pkl              # Serialized Scikit-learn Pipeline
├── app.py                     # FastAPI REST API with Pydantic validation
├── train.py                   # Data ingestion, preprocessing, training & export
├── test_app.py                # Automated Pytest suite for API & model logic
├── requirements.txt           # Pinned production and test dependencies
├── Dockerfile                 # Container build specification
└── .gitignore                 # Environment and cache ignores
