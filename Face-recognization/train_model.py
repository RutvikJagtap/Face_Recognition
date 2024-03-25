# USAGE
# python train_model.py --embeddings output/embeddings.pickle --recognizer output/recognizer.pickle --le output/le.pickle

# import the necessary packages
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import argparse
import pickle

# load the face embeddings
print("[INFO] loading face embeddings...")
data = pickle.loads(open("output/embeddings.pickle", "rb").read())

# encode the labels
print("[INFO] encoding labels...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])




param_grid = {
    'C': [0.1, 1, 10],  # C parameter values to try
    'kernel': ['linear', 'rbf'],  # Kernel types to try
    'gamma': ['scale', 'auto', 0.1, 1]  # Gamma values to try (for 'rbf' kernel)
}

# Create the SVM classifier
svm = SVC()

# Perform grid search with cross-validation to find the best parameters
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(data["embeddings"], labels)

# Print the best hyperparameters found by the grid search
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)




# train the model used to accept the 128-d embeddings of the face and
# then produce the actual face recognition
print("[INFO] training model...")
recognizer = SVC(C=best_params['C'], kernel=best_params['kernel'], gamma=best_params['gamma'],probability=True)
recognizer.fit(data["embeddings"], labels)

# write the actual face recognition model to disk
f = open("output/recognizer.pickle", "wb")
f.write(pickle.dumps(recognizer))
f.close()

# write the label encoder to disk
f = open("output/le.pickle", "wb")
f.write(pickle.dumps(le))
f.close()