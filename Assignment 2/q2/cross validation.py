import os
import cv2
import numpy as np
from sklearn.svm import SVC
import concurrent.futures
import matplotlib.pyplot as plt

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is not None:
            # Resize and normalize
            img_resized = cv2.resize(img, (16, 16)) / 255.0
            # Flatten
            img_flattened = img_resized.flatten()
            images.append(img_flattened)
    return np.array(images)

def multiclass_svm(X_train, y_train, C):
    clf = SVC(kernel='rbf', C=C, gamma=0.001)
    print(f"Training with C={C}")
    clf.fit(X_train, y_train)
    print("Done training")
    return clf

def train_and_evaluate(C, folds, X_train_shuffled, y_train_shuffled, X_val, y_val):
    accuracies = []
    for i, (start, end) in enumerate(folds):
        X_val_fold = X_train_shuffled[start:end]
        y_val_fold = y_train_shuffled[start:end]

        X_train_fold = np.concatenate([X_train_shuffled[:start], X_train_shuffled[end:]])
        y_train_fold = np.concatenate([y_train_shuffled[:start], y_train_shuffled[end:]])

        clf = multiclass_svm(X_train_fold, y_train_fold, C)
        accuracies.append(clf.score(X_val_fold, y_val_fold))

    cv_accuracy = np.mean(accuracies)
    clf = multiclass_svm(X_train_shuffled, y_train_shuffled, C)
    val_accuracy = clf.score(X_val, y_val)
    print(f"Result, C={C}, CV accuracy={cv_accuracy}, Validation accuracy={val_accuracy}")
    return cv_accuracy, val_accuracy

X_train = load_images_from_folder(f"../data/svm/train/0")
y_train = np.zeros(len(X_train))

for img_class in range(1, 6):
    class_images = load_images_from_folder(f"../data/svm/train/{img_class}")
    X_train = np.concatenate((X_train, class_images))
    y_train = np.concatenate((y_train, img_class * np.ones(len(class_images))))
# load all images for validation
X_val = load_images_from_folder(f"../data/svm/val/0")
y_val = np.zeros(len(X_val))

for img_class in range(1, 6):
    class_images = load_images_from_folder(f"../data/svm/val/{img_class}")
    X_val = np.concatenate((X_val, class_images))
    y_val = np.concatenate((y_val, img_class * np.ones(len(class_images))))

indices = list(range(len(X_train)))
np.random.shuffle(indices)

X_train_shuffled = X_train[indices]
y_train_shuffled = y_train[indices]
fold_size = len(X_train) // 5
folds = [(start, start + fold_size) for start in range(0, len(X_train), fold_size)]

C_values = [1e-5, 1e-3, 1, 5, 10]
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(train_and_evaluate, C, folds, X_train_shuffled, y_train_shuffled, X_val, y_val) for C in C_values]

results_with_C = list(zip(C_values, futures))
results_sorted = sorted(results_with_C, key=lambda x: x[0])

cv_accuracies = []
val_accuracies = []

for C, f in results_sorted:
    cv_acc, val_acc = f.result()
    cv_accuracies.append(cv_acc)
    val_accuracies.append(val_acc)


# Plotting
plt.plot(C_values, cv_accuracies, label="5-fold CV")
plt.plot(C_values, val_accuracies, label="Validation")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("cross_validation_mult real.png")
plt.show()
