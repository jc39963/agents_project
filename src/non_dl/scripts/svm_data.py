import kagglehub
from src.non_dl.utils import clahe_grayscale, add_padding


def extract_xml(path):
    CLASS_MAP = {
        "sunglass": 0,
        "hat": 1,
        "jacket": 2,
        "shirt": 3,
        "pants": 4,
        "shorts": 5,
        "skirt": 6,
        "dress": 7,
        "bag": 8,
        "shoe": 9,
    }
    tree = ET.parse(path)
    root = tree.getroot()
    all_objs = []
    for object in root.findall("object"):
        name = object.find("name").text
        box = object.find("bndbox")
        xmin = round(float(box.find("xmin").text))
        xmax = round(float(box.find("xmax").text))
        ymin = round(float(box.find("ymin").text))
        ymax = round(float(box.find("ymax").text))
        w = xmax - xmin
        h = ymax - ymin
        cx = (xmax + xmin) / 2
        cy = (ymin + ymax) / 2
        id = CLASS_MAP[name]
        all_objs.append([id, cx, cy, w, h])
    return np.array(all_objs)


# Function to visualize image with bounding boxes
def plot_bounding(img_path, bboxes):
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for bbox in bboxes:
        x0 = bbox[1] - bbox[3] / 2
        x1 = bbox[1] + bbox[3] / 2
        y0 = bbox[2] - bbox[4] / 2
        y1 = bbox[2] + bbox[4] / 2
        draw.rectangle([x0, y0, x1, y1], outline="red")
    display(img)


def save_images(img, bbox_list, base_img_id, split):
    counter = 0
    for bbox in bbox_list:
        counter += 1
        label = bbox[0]
        dir = f"data/crops/{split}/class_{label}"  # Saves to data directory under crops/train_OR_test/class_label
        if not os.path.exists(dir):
            os.makedirs(dir)
        x0 = int(bbox[1] - bbox[3] / 2)
        x1 = int(bbox[1] + bbox[3] / 2)
        y0 = int(bbox[2] - bbox[4] / 2)
        y1 = int(bbox[2] + bbox[4] / 2)
        height = y1 - y0
        width = x1 - x0
        cropped = img[y0:y1, x0:x1]
        cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        normalized = clahe_grayscale(cropped)
        padded = add_padding(normalized, height, width)
        final = cv2.resize(padded, (128, 128))
        cv2.imwrite(f"{dir}/{base_img_id}_{counter}.jpg", final)


def preprocess_images(list_of_ids, split):
    for i in list_of_ids:
        xml = i + ".xml"
        jpg_path = i + ".jpg"
        bboxes = extract_xml(os.path.join(xml_path, xml))
        img = cv2.imread(os.path.join(img_path, jpg_path))
        save_images(img, bboxes, i, split)


if __name__ == "__main__":
    path = kagglehub.dataset_download(
        "nguyngiabol/colorful-fashion-dataset-for-object-detection"
    )
    base_path = os.path.join(path, "colorful_fashion_dataset_for_object_detection")
    xml_path = os.path.join(base_path, "Annotations")
    img_path = os.path.join(base_path, "JPEGImages")
    trainval_path = os.path.join(base_path, "ImageSets/Main/trainval.txt")
    test_path = os.path.join(base_path, "ImageSets/Main/test.txt")
    with open(trainval_path, "r") as f:
        trainval_list = f.read().splitlines()
    with open(test_path, "r") as f:
        test_list = f.read().splitlines()
    preprocess_images(trainval_list, "train")
    preprocess_images(test_list, "test")
