# Dataset Description

## Dataset Used : IDD Detection (22.8 GB)

We used the IDD Detection dataset for object detection fine-tuning by INSAAN. The dataset is designed for Indian road scenes and contains annotations for common road objects such as pedestrians, riders, cars, trucks, buses, motorcycles, bicycles, autorickshaws, animals, traffic lights, and traffic signs.

The dataset was chosen because our project specifically targets navigation assistance in cluttered Indian road environments.

## Classes Used while fine-tuning

1. person
2. rider
3. car
4. truck
5. bus
6. motorcycle
7. bicycle
8. autorickshaw
9. animal
10. traffic light
11. traffic sign

## Train / Validation / Test Split

We used the predefined `train.txt`, `val.txt` and `test.txt` files provided in the dataset.

- Training images: 31,569 (67.76%)
- Validation images: 10,225 (21.95%)
- Test images: 4,794 (10.29%)

## Preprocessing

The original XML annotations were converted into YOLO format:
<class_id> <x_center> <y_center> <width> <height>

Images were linked into:

- images/train
- images/val

Labels were stored in:

- labels/train
- labels/val

## Input Resolution

All images were resized to 640 × 640 during training.

## Reduced Setup / Constraints

Due to compute limitations on Kaggle GPU sessions, we trained YOLOv8m for 25 epochs with batch size 16 instead of training for longer schedules such as 50–60 epochs.
