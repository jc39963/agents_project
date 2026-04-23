import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def compute_features(base_path):
  hog = cv2.HOGDescriptor(_winSize=(128, 128), _blockSize=(16, 16), _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9)
  feature_vecs = []
  labels = []
  for dir in os.listdir(base_path):
    label = dir.split('_')[1]
    dir = os.path.join(base_path, dir)
    for img_name in os.listdir(dir):
      img_name = os.path.join(dir, img_name)
      if os.path.isfile(img_name):
        img = cv2.imread(img_name)
        hog_feats = hog.compute(img)
        if np.all(hog_feats==0):
          print(f'Feature vector is all 0s!')
      else:
        print(f'{img_name} is not a valid file!')
      feature_vecs.append(hog_feats)
      labels.append(label)
  return np.vstack(feature_vecs), np.vstack(labels)

if __name__ == "__main__":
    # Prepare data features
    train_img_path = os.path.join('data', 'crops', 'train')
    test_img_path = os.path.join('data', 'crops', 'test')
    train_feats, train_labels = compute_features(train_img_path)
    test_feats, test_labels = compute_features(test_img_path)
    
    # PCA (reduces dimension to 1655)
    pca = PCA(n_components=0.95)
    pca.fit(train_feats)
    train_X = pca.transform(train_feats)
    test_X = pca.transform(test_feats)
    
    # Standardization of data before passing to SVM
    scaler = StandardScaler()
    scaler.fit(train_X)
    train_X = scaler.transform(train_X)
    test_X = scaler.transform(test_X)
    
    # Train SVM model
    svm = SVC(kernel='linear', C=0.1)
    svm.fit(train_X, train_labels.ravel())
    
    # Performance metrics on test dataset
    preds = svm.predict(test_X)
    print(f'Test Accuracy: {acuracy_score(test_labels, preds)}')
    print(f'Test F1 Score (macro): {f1_score(test_labels, preds, average="macro")}')
    
    # Plot test confusion matrix
    disp = ConfusionMatrixDisplay(cm, display_labels=['Sunglasses', 'Hat', 'Jacket', 'Shirt', 'Pants', 'Shorts', 'Skirt', 'Dress', 'Bag', 'Shoe'])
    disp.plot(values_format='.2f')
    plt.title('Test Confusion Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.show()
    
    # Save PCA, scaler, and SVM for later use
    joblib.dump(svm, 'src/non_dl/artifacts/svm_model.joblib')
    joblib.dump(pca, 'src/non_dl/artifacts/pca_model.joblib')
    joblib.dump(scaler, 'src/non_dl/artifacts/scaler_model.joblib')