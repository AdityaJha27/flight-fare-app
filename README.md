# ✈️ Flight Fare Prediction - End to End Machine Learning Project

## 📝 Overview
This is a high-performance Machine Learning web application that predicts flight ticket fares in India. The project uses an XGBoost Regressor optimized with Hyperparameter Tuning to estimate prices based on airlines, routes, and travel timings.

## 🚀 Live Demo
Check out the live application here: 
https://flight-fare-predictor-aditya.streamlit.app/

## 🖼️ Project Screenshot
![Project Screenshot](project_screenshot.png)

## ✨ Key Features
- **Hyperparameter Tuning:** Optimized using `RandomizedSearchCV` to achieve the best model performance and minimize error rates (MAE, MAPE).  
- **Advanced Feature Engineering:** Extracted critical insights from temporal data like Journey Day, Month, and Arrival/Departure timings.  
- **Smart Encoding:** Implemented One-Hot Encoding for categorical variables like Airlines, Source, and Destination.  
- **Real-time Prediction:** A responsive dashboard built with Streamlit for instant price estimation.  

## 🏗️ Technical Architecture
- **Data Preprocessing:** Handled missing values and transformed date-time strings into numeric features using Pandas.  
- **Model Training:** Utilized an XGBoost Regressor to capture complex non-linear relationships and boost accuracy in airfare data.  
- **Optimization:** Fine-tuned parameters like `learning_rate`, `max_depth`, and `n_estimators` for maximum accuracy.  
- **Deployment:** Serialized the model into `flight_xgb.pkl` and deployed via Streamlit Cloud.  

## 🛠️ Tech Stack & Technologies Used
- **Frontend:** Streamlit (Web Framework)  
- **Machine Learning:** XGBoost, Scikit-learn (`RandomizedSearchCV`)  
- **Data Manipulation:** Pandas, NumPy  
- **IDE:** VS Code & Google Colab  
- **Deployment:** Streamlit Cloud  

## 📁 Directory Tree
```
├── venv/                           # Virtual Environment Folder
├── .gitignore                      # Git Ignore File
├── app.py                          # Main Streamlit Application Backend
├── flight_fare_prediction.ipynb    # Model Training & Hyperparameter Tuning Notebook
├── flight_xgb.pkl                  # Tuned XGBoost Model (Joblib File)
├── project_screenshot.png          # Project Output Screenshot
├── README.md                       # Project Documentation
└── requirements.txt                # Project Dependencies
```
## 📊 Dataset
The model is trained on the popular Flight Fare dataset from Kaggle.  

- Dataset Link:
  https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh

## ⚙️ Installation
To run this project locally, clone the repository and install the dependencies:

```bash
git clone https://github.com/adityajha27/flight-fare-app.git
cd flight-fare-app
pip install -r requirements.txt
streamlit run app.py
```

Developed with ❤️ by **Aditya Jha**
---