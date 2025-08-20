🌸 Iris Classifier

A simple **Machine Learning project** that predicts the species of an Iris flower using a trained model.  
The project includes:
- A **trained ML model** (`iris_best_model.joblib`)
- A **CLI prediction script** (`predict.py`)
- A **Streamlit web app** (`app.py`)

---

## 📂 Project Structure
── models/
│ └── iris_best_model.joblib # Saved ML model
├── predict.py # CLI script for predictions
├── streamlit_app.py # Streamlit web app
├── README.md # Project documentation

---

## 🚀 Features
- Predict Iris flower species based on 4 input features:
  - Sepal Length
  - Sepal Width
  - Petal Length
  - Petal Width
- Two ways to use:
  - **Command Line Interface (CLI)**
  - **Streamlit Web Application**
- Shows prediction probabilities (if model supports `predict_proba`).

---

## ⚙️ Installation

1. Clone this repository:
   git clone https://github.com/your-username/iris-classifier.git
   cd iris-classifier
2.Install dependencies:
pip install -r requirements.txt
2) Streamlit Web App

Run:

python -m streamlit run streamlit_app.py


Then open the link (default: http://localhost:8501) in your browser.

🖥️ Usage
1) CLI Prediction

Run:
python predict.py 5.1 3.5 1.4 0.2
Output:
Predicted: setosa (class 0)
 setosa: 0.98
 versicolor: 0.01
 virginica: 0.01
📊 Example UI

The Streamlit app provides sliders to input flower measurements and outputs:

Predicted species

Probability distribution across classes

🛠️ Built With

Scikit-learn
 – ML model training

Joblib
 – Model saving/loading

Streamlit
 – Web app interface

NumPy
 – Numerical operations

✨ Future Improvements

Deploy app to Streamlit Cloud or Heroku

Add data visualization

Train with more datasets for robustness

📜 License

This project is licensed under the MIT License – feel free to use and modify it.


