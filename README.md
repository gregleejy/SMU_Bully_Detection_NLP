# SMU_Bully_Detection_NLP

SMU Confess Bullying Detection - Sentiment Analysis Project

Project Overview

This project detects targeted bullying remarks in the Telegram group chat SMU Confess using Natural Language Processing (NLP). The model identifies harmful messages through sentiment analysis and flags messages that could be considered bullying. The goal is to create a safer online space using AI-powered text classification.

Table of Contents

	1.	Project Overview
	2.	Motivation
	3.	Technologies Used
	4.	Installation
	5.	Project Structure
	6.	Model Development
	7.	Usage
	8.	Future Improvements
	9.	Contributing
	10.	License

Motivation

Online bullying and harassment are serious concerns in anonymous group chats. SMU Confess, being a platform for anonymous sharing, can sometimes host harmful messages. This project was built to identify and flag those messages using sentiment analysis, ensuring a safer environment.

Technologies Used

	•	Programming Language: Python
	•	Libraries/Frameworks:
	•	NLP: Hugging Face Transformers, NLTK, spaCy
	•	Machine Learning: scikit-learn, TensorFlow, PyTorch
	•	Data Handling: Pandas, NumPy
	•	Web Scraping: Telethon API
	•	Deployment: Streamlit, Flask, Heroku

Installation

	1.	Clone the repository:

git clone https://github.com/yourusername/SMU_Bully_Detection_NLP.git
cd SMU_Bully_Detection_NLP


	2.	Create a virtual environment and activate it:

python -m venv env
source env/bin/activate  # Linux/Mac
env\Scripts\activate  # Windows


	3.	Install dependencies:

pip install -r requirements.txt

Project Structure

SMU_Bully_Detection_NLP/
├── data/                  # Data files (CSV, JSON, etc.)
├── models/                # Trained models
├── notebooks/             # Jupyter notebooks for exploration
├── src/                   # Core project scripts
│   ├── data_preprocessing.py
│   ├── train_model.py
│   ├── evaluate_model.py
├── docs/                  # Documentation files
├── README.md              # Project overview
├── requirements.txt       # Required libraries
├── app.py                 # Deployment script (Streamlit or Flask)
└── LICENSE                # License file

Model Development

	1.	Data Collection: Extract Telegram chat data using the Telethon API.
	2.	Data Preprocessing: Clean, tokenize, and prepare the data.
	3.	Model Training: Train models such as BERT or DistilBERT.
	4.	Evaluation: Evaluate the model using metrics such as accuracy, F1-score, precision, and recall.

Usage

	•	Run the app locally:

streamlit run app.py


	•	Example command-line prediction:

from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")
result = sentiment_model("You are such a loser!")
print(result)

Future Improvements

	•	Fine-tune the BERT model on a custom bullying detection dataset.
	•	Add multi-language support for global user interaction.
	•	Improve accuracy using ensemble learning methods.

Contributing

Contributions are welcome! If you’d like to help improve this project, please create a pull request or contact me directly.

License

This project is licensed under the MIT License. See the LICENSE file for more details.
