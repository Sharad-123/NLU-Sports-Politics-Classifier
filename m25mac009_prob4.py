
# PROBLEM 4: SPORTS vs POLITICS CLASSIFIER


# Author: Sharad Kumar Singh
# Roll Number: Your Roll Number



import os
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report



# Loading the dataset

def load_data(dataset_path):

    texts = []
    labels = []

    # Two categories
    categories = ["sports", "politics"]

    for category in categories:

        folder_path = os.path.join(dataset_path, category)

        if not os.path.exists(folder_path):
            print("Error: Folder not found ", folder_path)
            continue

        for filename in os.listdir(folder_path):

            if filename.endswith(".txt"):

                file_path = os.path.join(folder_path, filename)

                try:
                    with open(file_path, "r", encoding="utf-8") as file:

                        content = file.read().strip()

                        # Ignore empty or very small files
                        if len(content) > 10:
                            texts.append(content)
                            labels.append(category)

                except:
                    print("Error reading file:", file_path)

    return texts, labels



# Main Function


def main():

    # Change this if needed
    DATASET_PATH = "dataset"

    print("Loading dataset...")

    texts, labels = load_data(DATASET_PATH)

    print("Total documents loaded:", len(texts))

    if len(texts) == 0:
        print("No data found. Please check dataset folder.")
        return


    
    # TF-IDF feature extraction technique


    print("\nExtracting TF-IDF features")

    vectorizer = TfidfVectorizer(

        lowercase=True,

        stop_words="english",

        max_features=1000,

        ngram_range=(1, 1),

        min_df=2,

        max_df=0.9
    )

    X = vectorizer.fit_transform(texts)

    y = np.array(labels)


    
    # Train test split 70 30


    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.3,

        stratify=y,

        random_state=42
    )

    print("Training samples:", X_train.shape[0])
    print("Testing samples:", X_test.shape[0])


    
    # Define Models
    

    models = {

        "Naive Bayes":
            MultinomialNB(alpha=1.0),

        "Logistic Regression":
            LogisticRegression(max_iter=1000, C=0.5),

        "Support Vector Machine":
            LinearSVC(C=0.5)
    }



    # Training and evaluating each model
    

    for model_name in models:

        model = models[model_name]

        print("\n===================================")
        print(model_name)
        print("===================================")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        print("Test Accuracy:", round(accuracy, 4))

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))


        
        # Cross Validation accuracy
        

        cv_scores = cross_val_score(

            model,

            X,

            y,

            cv=5
        )

        print("Cross Validation Accuracy:", round(cv_scores.mean(), 4))




if __name__ == "__main__":

    main()
