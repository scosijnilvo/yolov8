## Installation
Clone the repository
```bash
git clone https://github.com/scosijnilvo/yolov8
cd yolov8
```

Create and activate a virtual env
```bash
python -m venv /path/to/venv
source /path/to/venv/bin/activate
```

Install with pip
```bash
pip install .
```
(or in editable mode if you want to modify the code)
```bash
pip install -e .
```

## Dataset
Create ```dataset.yaml``` (edit paths, classes, and num_vars to fit your dataset)
```yaml
path: ../yolov8/ # root dir
train: dataset/images/train # train images
val: dataset/images/val # val images
test: dataset/images/test # test images (optional)

# Classes
names:
  0: class_0
  1: class_1
  # ...

# Number of variables to predict
num_vars: 1
```

Each image must have a corresponding label file with the same name and ```.txt``` extension located at ```dataset/labels/[train|val|test]```.
The label files consist of one row for each object in the image with the following format.

For detection:
```
<class-index> <var_1> ... <var_n> <x> <y> <w> <h>
```

For segmentation:
```
<class-index> <var_1> ... <var_n> <x1> <y1> ... <xn> <yn>
```

where
- ```<class-index>``` = index of the class declared in the ```.yaml``` file
- ```<var_1> ... <var_n>``` = ground-truth values of the variables, set ```num_vars``` in the ```.yaml``` file to ```n```
- ```<x> <y> <w> <h>``` = bounding box coordinates in xywh-format, normalized between 0 and 1
- ```<x1> <y1> ... <xn> <yn>``` = bounding coordinates of the segmentation mask, normalized between 0 and 1

## Example code
```python
# import
from ultralytics import RegressionModel

# training a detection model
model = RegressionModel('yolov8s-det-regression.yaml')
results = model.train(data='dataset.yaml', epochs=100)

# training a segmentation model
model = RegressionModel('yolov8s-seg-regression.yaml')
results = model.train(data='dataset.yaml', epochs=100)

# loading + evaluating on test set
model = RegressionModel('saved_model.pt')
metrics = model.val(split='test')
```

## CLI
### Training
Detection
```
yolo train model=yolov8s-det-regression.yaml data=dataset.yaml epochs=100
```
Segmentation
```
yolo train model=yolov8s-seg-regression.yaml data=dataset.yaml epochs=100
```

### Validation
```
yolo val model=saved_model.pt data=dataset.yaml
```

### Inference
```
yolo predict model=saved_model.pt source=image.jpg
```
