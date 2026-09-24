# 🚦 ATX Traffic Incident Analysis (2023-2026)

## Overview
An end-to-end data analytics and machine learning pipeline analyzing over 310,000 real-time traffic incidents in Austin, TX. This project evolved from basic exploratory data analysis to a robust pipeline featuring statistical hypothesis testing, geospatial clustering, and predictive modeling.

## 🛠 Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (K-Means Clustering, Logistic Regression)
* **Statistics:** SciPy (ANOVA, Chi-Square tests)
* **Visualization:** Matplotlib, Seaborn

## 📊 Key Highlights
1. **Data Engineering:** Built a robust cleaning pipeline to standardize messy categorical data, validate geographic coordinates, and engineer 40+ temporal features.
2. **Statistical Rigor:** 
   * Conducted **ANOVA testing** to prove incident volumes differ significantly by hour of the day.
   * Conducted **Chi-Square testing** to confirm incident type proportions changed significantly YoY.
3. **Geospatial Analysis:** Applied **K-Means Clustering** to segment Austin into localized high-risk traffic zones to optimize theoretical dispatch routing.
4. **Predictive Modeling:** Trained a **Logistic Regression** classifier (with Train/Test splits and StandardScaler) to predict incident urgency.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Download the raw data from the [City of Austin Open Data Portal](https://data.austintexas.gov/Transportation-and-Mobility/Real-Time-Traffic-Incident-Reports/dx9v-zd7x/data_preview) and place it in `data/raw/`.
4. Run the pipeline:
   * `python src/clean_transform.py`
   * `python src/feature_engineering.py`
   * `python src/temporal_analysis.py`
   * `python src/geospatial_analysis.py`
   * `python src/modeling.py`
