# Dataset
## Containing files

### fullbody_train_charm.csv/fullbody_test_charm.csv
Made from [Jan Bre's notebook](https://www.kaggle.com/code/jpbremer/backfin-detection-with-yolov5) with DIM=1024, EPOCH=50.
We used fullbody_annotations.csv from [Jan Bre's dataset](https://www.kaggle.com/datasets/jpbremer/fullbodywhaleannotations) as training data.

### species.npy/individual_id.npy
Arrays of label encoders used in charmq's pipeline.

### pseudo_labels/
List of pseudo labels we created in each round.


## Files you need to download from Kaggle
### train.csv/sample_submission.csv/train_images/test_images
[Competition dataset](https://www.kaggle.com/competitions/happy-whale-and-dolphin/data).

### fullbody_train.csv/fullbody_test.csv
[Jan Bre's dataset](https://www.kaggle.com/datasets/jpbremer/fullbodywhaleannotations).

### train_backfin.csv/test_backfin.csv
[Jan Bre's notebook](https://www.kaggle.com/code/jpbremer/backfin-detection-with-yolov5) (copy of train.csv/test.csv).

### train2.csv/test2.csv
[phalanx's dataset](https://www.kaggle.com/datasets/phalanx/whale2-cropped-dataset). Bboxes of Detic.

### yolov5_train.csv/yolov5_test.csv
[Awsaf's notebook](https://www.kaggle.com/code/awsaf49/happywhale-cropped-dataset-yolov5) (copy of train.csv/test.csv).
