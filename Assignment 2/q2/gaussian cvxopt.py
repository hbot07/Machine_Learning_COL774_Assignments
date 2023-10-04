import numpy as np
from cvxopt import matrix, solvers
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score
import cv2
import os

# Load data
def load_data(path):
    X, y = [], []
    for label, folder_name in enumerate(['0', '5']):
        folder_path = os.path.join(path, folder_name)
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)
            img_resized = cv2.resize(img, (16, 16))
            img_flattened = img_resized.flatten() / 255.0
            X.append(img_flattened)
            y.append(1 if label == 0 else -1)
    return np.array(X), np.array(y)

X, y = load_data('../data/svm/train')
X_val, y_val = load_data('../data/svm/val')

# Gaussian kernel function
def gaussian_kernel(x1, x2, gamma=0.001):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)

# Compute P matrix using Gaussian kernel
squared_dists = cdist(X, X, 'sqeuclidean')
gamma = 0.001
K = np.exp(-gamma * squared_dists)
P = np.outer(y, y) * K

# Define the other matrices for CVXOPT
q = -np.ones((X.shape[0], 1))
G = np.vstack([-np.eye(X.shape[0]), np.eye(X.shape[0])])
h = np.hstack([np.zeros(X.shape[0]), np.ones(X.shape[0]) * 1.0])
A = y.reshape(1, -1).astype(float)
b = np.zeros(1)

# Convert to CVXOPT format and solve
P_cvxopt = matrix(P)
q_cvxopt = matrix(q)
G_cvxopt = matrix(G)
h_cvxopt = matrix(h)
A_cvxopt = matrix(A)
b_cvxopt = matrix(b)

sol = solvers.qp(P_cvxopt, q_cvxopt, G_cvxopt, h_cvxopt, A_cvxopt, b_cvxopt)
alphas = np.array(sol['x'])

# Compute b using a support vector
sv_indices = (alphas > 1e-5).flatten()
support_vectors = X[sv_indices]
support_vector_labels = y[sv_indices]
alphas_sv = alphas[sv_indices]

b = support_vector_labels[0] - np.sum(alphas_sv * support_vector_labels *
                                      np.array([gaussian_kernel(support_vectors[0], sv) for sv in support_vectors]))

# Predict function
def predict(X):
    y_pred = []
    for x in X:
        prediction = sum(alphas[i] * y[i] * gaussian_kernel(X[i], x)
                         for i in range(len(X))) + b
        y_pred.append(np.sign(prediction))
    return np.array(y_pred)

# Calculate validation accuracy
y_val_pred = predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print("Validation accuracy with Gaussian kernel:", val_accuracy)