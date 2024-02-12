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
Create ```dataset.yaml``` (edit paths and classes to fit your dataset)
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
```

Each image must have a corresponding label file with the same name and ```.txt``` extension located at ```dataset/labels/[train|val|test]```.
The label files consist of one row for each object in the image with the following format.

For detection:
```
<class-index> <weight> <x> <y> <w> <h>
```

For segmentation:
```
<class-index> <weight> <x1> <y1> ... <xn> <yn>
```

where
- ```<class-index>``` = index of the class declared in the ```.yaml``` file
- ```<weight>``` = weight of the object
- ```<x> <y> <w> <h>``` = bounding box coordinates in xywh-format, normalized between 0 and 1
- ```<x1> <y1> ... <xn> <yn>``` = bounding coordinates of the segmentation mask, normalized between 0 and 1

## Example code
```python
# import
from ultralytics.models import WeightModel

# training a detection model
model = WeightModel('yolov8n-weight.yaml', task='detect')
results = model.train(data='dataset.yaml', epochs=100)

# training a segmentation model
model = WeightModel('yolov8n-weight-seg.yaml', task='segment')
results = model.train(data='dataset.yaml', epochs=100)

# loading + evaluating on test set
model = WeightModel('saved_model.pt', task='segment')
metrics = model.val(split='test')
```