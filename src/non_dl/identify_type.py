from .utils import *


def identify_type(
    img_path: str = "data/images/captured.jpg",
) -> tuple[str, list[float]]:
    """Uses a pre-trained SVM to classify the clothing item in an image into 1 of 10 categories.

    Args:
        img_path (str, optional): Path to image of clothing item to be identified. Defaults to 'data/images/captured.jpg'.

    Returns:
        str: Category label for the clothing item.
        list[float]: Feature vector for clothing item in image, later used for catalog search to find similar style items.
    """
    REVERSE_CLASS_MAP = {
        0: "sunglass",
        1: "hat",
        2: "jacket",
        3: "shirt",
        4: "pants",
        5: "shorts",
        6: "skirt",
        7: "dress",
        8: "bag",
        9: "shoe",
    }
    svm, pca, scaler = load_artifacts()
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    normalized = clahe_grayscale(img)
    padded = add_padding(normalized, height, width)
    final = cv2.resize(padded, (128, 128))
    hog = cv2.HOGDescriptor(
        _winSize=(128, 128),
        _blockSize=(16, 16),
        _blockStride=(8, 8),
        _cellSize=(8, 8),
        _nbins=9,
    )
    hog_features = hog.compute(final).reshape(1, -1)
    post_pca = pca.transform(hog_features)
    post_scaling = scaler.transform(post_pca)
    pred = float(svm.predict(post_scaling)[0])
    final_feat_vec = post_scaling.flatten().tolist()
    return REVERSE_CLASS_MAP[int(pred)], final_feat_vec


# print(identify_type("data/images/clothing_roi_test.jpg"))
