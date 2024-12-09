import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, log_loss
import os

# SVM modeli eğitimi ve kaydedilmesi için kullanılan kod
DATA_DIR = "/Users/zehraatalay/Desktop/dataset"

# Pickle dosyalarını yükleme
train_data_path = os.path.join(DATA_DIR, "train_data.pickle")
validation_data_path = os.path.join(DATA_DIR, "validation_data.pickle")
test_data_path = os.path.join(DATA_DIR, "test_data.pickle")

train_data = pickle.load(open(train_data_path, "rb"))
validation_data = pickle.load(open(validation_data_path, "rb"))
test_data = pickle.load(open(test_data_path, "rb"))

# Tüm veri setleri için uyumlu ve uyumsuz verileri kontrol eden bir fonksiyon
def filter_and_explain(data_dict, expected_length):
    filtered_data_labels = []
    incompatible_data = []

    for idx, (data, label) in enumerate(zip(data_dict["data"], data_dict["labels"])):
        if len(data) == expected_length:
            filtered_data_labels.append((data, label))
        else:
            incompatible_data.append((idx, len(data), label))

    return filtered_data_labels, incompatible_data


# Train verisi için işlemler
expected_length = len(train_data["data"][0])
train_filtered, train_incompatible = filter_and_explain(train_data, expected_length)
xtrain, ytrain = zip(*train_filtered) if train_filtered else ([], [])
xtrain = np.array(xtrain)
ytrain = np.array(ytrain)

# Validation verisi için işlemler
val_filtered, val_incompatible = filter_and_explain(validation_data, expected_length)
xval, yval = zip(*val_filtered) if val_filtered else ([], [])
xval = np.array(xval)
yval = np.array(yval)

# Test verisi için işlemler
test_filtered, test_incompatible = filter_and_explain(test_data, expected_length)
xtest, ytest = zip(*test_filtered) if test_filtered else ([], [])
xtest = np.array(xtest)
ytest = np.array(ytest)

# Uyumlu ve uyumsuz veri detaylarını yazdırma
print(f"Train: {len(train_filtered)} compatible, {len(train_incompatible)} incompatible")
if train_incompatible:
    print(f"Incompatible Train Data: {train_incompatible}")

print(f"Validation: {len(val_filtered)} compatible, {len(val_incompatible)} incompatible")
if val_incompatible:
    print(f"Incompatible Validation Data: {val_incompatible}")

print(f"Test: {len(test_filtered)} compatible, {len(test_incompatible)} incompatible")
if test_incompatible:
    print(f"Incompatible Test Data: {test_incompatible}")

# Model oluşturma ve eğitme
if len(train_filtered) > 0:
    model = SVC(probability=True)  # SVM modeli, olasılık hesaplaması için `probability=True`
    model.fit(xtrain, ytrain)

    # Train doğruluğu ve kaybı
    ytrain_pred = model.predict(xtrain)
    ytrain_proba = model.predict_proba(xtrain)
    train_accuracy = accuracy_score(ytrain, ytrain_pred)
    train_loss = log_loss(ytrain, ytrain_proba)
    print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
    print(f"Train Loss: {train_loss:.4f}")

    # Validation doğruluğu ve kaybı
    if len(val_filtered) > 0:
        yval_pred = model.predict(xval)
        yval_proba = model.predict_proba(xval)
        val_accuracy = accuracy_score(yval, yval_pred)
        val_loss = log_loss(yval, yval_proba)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
    else:
        print("No compatible validation data.")

    # Test doğruluğu ve kaybı
    if len(test_filtered) > 0:
        ytest_pred = model.predict(xtest)
        ytest_proba = model.predict_proba(xtest)
        test_accuracy = accuracy_score(ytest, ytest_pred)
        test_loss = log_loss(ytest, ytest_proba)
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
    else:
        print("No compatible test data.")

    # Modeli kaydetme
    model_path = os.path.join(DATA_DIR, "svm_model.p")
    with open(model_path, "wb") as f:
        pickle.dump({"model": model}, f)
    print(f"Model successfully saved to {model_path}")

else:
    print("No compatible training data. Model training aborted.")
