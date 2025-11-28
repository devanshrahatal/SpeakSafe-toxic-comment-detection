import pickle
from preprocess import clean_text

# Load model + vectorizer
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))
model = pickle.load(open("model/svm_model.pkl", "rb"))

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

print("\n🔍 Toxic Comment Detector (Type 'exit' to quit)\n")

while True:
    text = input("Enter a comment: ")

    if text.lower() == "exit":
        break

    # preprocess
    cleaned = clean_text(text)

    # vectorize
    vector = vectorizer.transform([cleaned])

    # predict
    pred = model.predict(vector)[0]

    print("\nPrediction:")
    for label, value in zip(labels, pred):
        if value == 1:
            print(f"✔ {label}")
    if pred.sum() == 0:
        print("✔ Non-toxic")

    print("\n" + "-"*40 + "\n")
