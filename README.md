# IBM_Cloud_Project
Power System Fault Detection and Classification
A machine learning-based solution for automatic detection and classification of power system faults using IBM Watson AI and cloud services.
📋 Project Overview
This project addresses the critical challenge of rapid fault identification in electrical distribution networks. Using electrical measurement data (voltage and current phasors), the system can distinguish between normal operating conditions and various fault conditions to maintain power grid stability and reliability.
🎯 Problem Statement
Power distribution systems require continuous monitoring to ensure reliable electricity supply. Traditional fault detection methods are often slow and may not accurately classify fault types, leading to:

Extended power outages
Equipment damage
Grid instability
Economic losses

🚀 Features

Real-time Fault Detection: Automatic identification of power system anomalies
Multi-class Classification: Distinguishes between different fault types (line-to-ground, line-to-line, three-phase)
Cloud-based Deployment: Scalable solution using IBM Cloud services
High Accuracy: Machine learning models optimized for power system data
RESTful API: Easy integration with existing monitoring systems

🛠️ Technology Stack

Cloud Platform: IBM Cloud Lite Services
AI/ML Services: IBM Watson AI, Watson Machine Learning
Programming Language: Python
Libraries:

pandas, numpy (Data processing)
scikit-learn (Machine learning)
matplotlib, seaborn (Visualization)
ibm-watson-machine-learning (IBM ML services)


Dataset: Power System Faults Dataset

📊 Dataset
The project uses electrical measurement data including:

Voltage phasor measurements (magnitude and phase angle)
Current phasor measurements (magnitude and phase angle)
System frequency measurements
Time-stamped electrical parameters
Historical fault occurrence data

🔧 Installation


Set up IBM Cloud credentials
bash# Create .env file and add your IBM Cloud credentials
IBM_CLOUD_API_KEY=your_api_key_here
IBM_WATSON_ML_URL=your_watson_ml_url

Download dataset

Download the dataset from Kaggle:https://www.kaggle.com/datasets/ziya07/power-system-faults




📈 Usage
Training the Model
python# Run the training script
python train_model.py

# Or use Jupyter notebook
jupyter notebook notebooks/model_training.ipynb
Making Predictions
pythonfrom src.predictor import FaultPredictor

# Initialize predictor
predictor = FaultPredictor()

# Load your electrical measurement data
data = load_data('path/to/your/data.csv')

# Predict fault type
prediction = predictor.predict(data)
print(f"Fault Type: {prediction['fault_type']}")
print(f"Confidence: {prediction['confidence']:.2f}")
API Usage
bash# Start the API server
python app.py

# Make predictions via REST API
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"voltage": [1.0, 0.0], "current": [0.5, 0.2]}'
📁 Project Structure
power-system-fault-detection/
├── data/                          # Dataset files
├── notebooks/                     # Jupyter notebooks
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   └── model_evaluation.ipynb
├── src/                          # Source code
│   ├── data_preprocessing.py
│   ├── model_training.py
│   ├── predictor.py
│   └── utils.py
├── models/                       # Trained models
├── static/                       # Web interface files
├── templates/                    # HTML templates
├── tests/                        # Unit tests
├── app.py                        # Flask API application
├── train_model.py               # Training script
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
└── .env.example                 # Environment variables template

Note: Results may vary based on dataset and hyperparameter tuning
📋 Algorithm Details
Models Evaluated

Random Forest: Ensemble method for robust classification
Support Vector Machine (SVM): Effective for high-dimensional data
Neural Networks: Deep learning approach for complex patterns

Training Process

Dataset split: 70% training, 15% validation, 15% testing
Feature scaling and normalization
Cross-validation for model robustness
Hyperparameter tuning using grid search
Model deployment on IBM Watson ML

🚀 Deployment
The model is deployed on IBM Cloud using Watson Machine Learning services:

IBM Watson Studio: Data preprocessing and model development
Watson Machine Learning: Model training and deployment
IBM Cloud Functions: Serverless API endpoints
Cloud Object Storage: Dataset and model storage

🔮 Future Enhancements

 Integration with IoT sensors for real-time monitoring
 Mobile application for field technicians
 Predictive maintenance capabilities
 Multi-region deployment support
 Integration with SCADA systems
 Edge computing implementation


College: Dronacharya Group of Institutions
Department: Computer Science Engineering
Email:shivamuniyal12345@gamil.com


📚 References

IBM Watson Machine Learning Documentation
Power System Fault Analysis Research Papers
Scikit-learn Documentation


