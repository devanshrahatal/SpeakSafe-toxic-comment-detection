# Toxic Comment Detection

This is a **Multi-label Toxic Comment Detection** project using Machine Learning and NLP.  
The system can classify a comment into one or more of the following categories:

- Toxic
- Severe Toxic
- Obscene
- Threat
- Insult
- Identity Hate

The project uses **TF-IDF + SVM (One-vs-Rest)** for fast and effective predictions.  
A modern **Streamlit UI** allows users to test comments in real-time.

---

## Project Structure

toxic-comment-detection/
│── model/
│ ├── tfidf_vectorizer.pkl
│ └── svm_model.pkl
│
│── preprocess.py
│── train.py
│── app.py
│── requirements.txt
│── README.md

- `model/` → Contains trained models  
- `preprocess.py` → Text cleaning functions  
- `train.py` → Train TF-IDF + SVM model  
- `app.py` → Streamlit UI for real-time prediction  

---

## Installation

- Install required packages:

pip install pandas numpy scikit-learn nltk streamlit
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

## Usage

1. Train the model (Optional)

If you want to retrain:
python train.py
This will save the trained model in model/ folder.

2. Run the Streamlit App

streamlit run app.py
- Enter a comment in the text box.
- Click Analyze.
- The app will show which toxic labels apply.

3. Test in Terminal (Optional)

You can also test comments in terminal:
python test_model.py
- Type a comment and see the predicted labels.
- Type exit to quit.

## Dataset

This project uses the Kaggle Jigsaw Toxic Comment Classification Challenge dataset.

## Future Improvements

- Use BERT/DistilBERT for higher accuracy
- Display toxicity percentages per label
- Add multi-language support
- Deploy as a web app with hosting

## Author
- Made with PREM by Devansh Rahatal
