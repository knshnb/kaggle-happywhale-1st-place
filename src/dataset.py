import albumentations as A
import cv2
import numpy as np
import pandas as pd
from albumentations.pytorch import ToTensorV2
from sklearn import preprocessing
from torch.utils.data import Dataset

from config.config import Config


class WhaleDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg: Config,
        image_dir: str,
        val_bbox_name: str,
        data_aug: bool,
    ):
        super().__init__()
        self.index = df.index
        self.x_paths = np.array(df.image)
        self.ids = np.array(df.individual_id, dtype=int) if hasattr(df, "individual_id") else np.full(len(df), -1)
        self.species = np.array(df.species, dtype=int) if hasattr(df, "species") else np.full(len(df), -1)
        self.cfg = cfg
        self.image_dir = image_dir
        self.df = df
        self.val_bbox_name = val_bbox_name
        self.data_aug = data_aug
        augments = []
        if data_aug:
            aug = cfg.aug
            augments = [
                A.Affine(
                    rotate=(-aug.rotate, aug.rotate),
                    translate_percent=(0.0, aug.translate),
                    shear=(-aug.shear, aug.shear),
                    p=aug.p_affine,
                ),
                A.RandomResizedCrop(
                    self.cfg.image_size[0],
                    self.cfg.image_size[1],
                    scale=(aug.crop_scale, 1.0),
                    ratio=(aug.crop_l, aug.crop_r),
                ),
                A.ToGray(p=aug.p_gray),
                A.GaussianBlur(blur_limit=(3, 7), p=aug.p_blur),
                A.GaussNoise(p=aug.p_noise),
                A.Downscale(scale_min=0.5, scale_max=0.5, p=aug.p_downscale),
                A.RandomGridShuffle(grid=(2, 2), p=aug.p_shuffle),
                A.Posterize(p=aug.p_posterize),
                A.RandomBrightnessContrast(p=aug.p_bright_contrast),
                A.Cutout(p=aug.p_cutout),
                A.RandomSnow(p=aug.p_snow),
                A.RandomRain(p=aug.p_rain),
                A.HorizontalFlip(p=0.5),
            ]
        augments.append(A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        augments.append(ToTensorV2())  # HWC to CHW
        self.transform = A.Compose(augments)

    def __len__(self):
        return len(self.ids)

    def get_original_image(self, i: int):
        bgr = cv2.imread(f"{self.image_dir}/{self.x_paths[i]}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def __getitem__(self, i: int):
        image = self.get_original_image(i)
        # crop
        if self.data_aug:
            bbox_name = np.random.choice(list(self.cfg.bboxes.keys()), p=list(self.cfg.bboxes.values()))
        else:
            bbox_name = self.val_bbox_name
        bbox = None if bbox_name == "none" else self.df[bbox_name].iloc[i]
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            image = image[ymin:ymax, xmin:xmax]
        # resize
        image = cv2.resize(image, self.cfg.image_size, interpolation=cv2.INTER_CUBIC)
        # data augmentation
        augmented = self.transform(image=image)["image"]
        return {
            "original_index": self.index[i],
            "image": augmented,
            "label": self.ids[i],
            "label_species": self.species[i],
        }


def load_bbox(cfg: Config, in_base_dir: str, bbox_name: str, is_train: bool) -> pd.Series:
    if bbox_name == "detic":
        filename = "train2.csv" if is_train else "test2.csv"
        tmp_df = pd.read_csv(f"{in_base_dir}/{filename}")
        low_conf = pd.Series([False for _ in range(len(tmp_df))])
        bbox = tmp_df.box.map(lambda s: list(map(int, s.split())) if s == s else None)
    elif bbox_name == "fullbody":
        filename = "fullbody_train.csv" if is_train else "fullbody_test.csv"
        tmp_df = pd.read_csv(f"{in_base_dir}/{filename}")
        low_conf = tmp_df.conf.map(lambda s: float(s[1:-1]) if s == s else -1) < cfg.bbox_conf_threshold
        bbox = tmp_df.bbox.map(lambda s: list(map(int, s[2:-2].split())))
    elif bbox_name == "fullbody_charm":
        filename = "fullbody_train_charm.csv" if is_train else "fullbody_test_charm.csv"
        tmp_df = pd.read_csv(f"{in_base_dir}/{filename}")
        low_conf = tmp_df.conf.map(lambda s: float(s[1:-1]) if s == s else -1) < cfg.bbox_conf_threshold
        bbox = tmp_df.bbox.map(lambda s: list(map(int, s[2:-2].split())) if s == s else None)
    elif bbox_name == "backfin":
        filename = "train_backfin.csv" if is_train else "test_backfin.csv"
        tmp_df = pd.read_csv(f"{in_base_dir}/{filename}")
        low_conf = tmp_df.conf.map(lambda s: float(s[1:-1]) if s == s else -1) < cfg.bbox_conf_threshold
        bbox = tmp_df.bbox.map(lambda s: list(map(int, s[2:-2].split())) if s == s else None)
    else:
        raise AssertionError()
    print(f"{bbox_name} low conf: {low_conf.sum()} / {len(tmp_df)}")
    bbox[low_conf] = None
    return bbox


def load_df(in_base_dir: str, cfg: Config, filename: str, is_train: bool) -> pd.DataFrame:
    df = pd.read_csv(f"{in_base_dir}/{filename}")

    # bbox
    for bbox_name in ["detic", "fullbody", "fullbody_charm", "backfin"]:
        df[bbox_name] = load_bbox(cfg, in_base_dir, bbox_name, is_train)

    # label encoder
    if hasattr(df, "individual_id"):
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.classes_ = np.load(f"{in_base_dir}/individual_id.npy", allow_pickle=True)
        df.individual_id = label_encoder.transform(df.individual_id)
        assert cfg.num_classes == len(label_encoder.classes_)
    if hasattr(df, "species"):
        df.species.replace(
            {
                "globis": "short_finned_pilot_whale",
                "pilot_whale": "short_finned_pilot_whale",
                "kiler_whale": "killer_whale",
                "bottlenose_dolpin": "bottlenose_dolphin",
            },
            inplace=True,
        )  # https://www.kaggle.com/c/happy-whale-and-dolphin/discussion/305574
        label_encoder_species = preprocessing.LabelEncoder()
        label_encoder_species.classes_ = np.load(f"{in_base_dir}/species.npy", allow_pickle=True)
        df.species = label_encoder_species.transform(df.species)
        assert cfg.num_species_classes == len(label_encoder_species.classes_)
    return df
