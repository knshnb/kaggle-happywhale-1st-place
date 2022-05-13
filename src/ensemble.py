import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse
import torch
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors

from config.config import Config, load_config


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make submission csv for HappyWhale")
    parser.add_argument("--in_base_dir", default="input")
    parser.add_argument("--model_dirs", nargs="+", required=True)
    parser.add_argument("--out_prefix", default="test")
    return parser.parse_args()


def load_results(results_path: str):
    results = np.load(results_path)
    # reorder all values so that ret["original_index"] = [0, 1, 2, ..., n_data - 1]
    n_data = results["original_index"].max() + 1
    ord = np.full(n_data, -1, dtype=int)
    ord[results["original_index"]] = np.arange(len(results["original_index"]))
    ret = {key: results[key][ord] for key in results.files if key != "file_name"}
    assert np.array_equal(ret["original_index"], np.arange(n_data))
    return ret


def restore_all_pred(n_class: int, pred: np.ndarray, pred_idx: np.ndarray):
    n_data = pred.shape[0]
    all_pred = np.zeros((n_data, n_class))
    for i in range(n_data):
        all_pred[i][pred_idx[i]] = pred[i]
    return all_pred


def knn_all_pred(n_class: int, n_train: int, test_feat: np.ndarray, train_feat: np.ndarray, train_label: np.ndarray):
    neigh = NearestNeighbors(n_neighbors=500, metric="cosine")
    neigh.fit(train_feat)
    test_dist, test_cosine_idx = neigh.kneighbors(test_feat, return_distance=True)  # [n_val, 1000], [n_val, 1000]
    test_cosine = 1 - test_dist
    test_cosine_idx %= n_train
    test_all_knn = np.zeros((len(test_feat), n_class))
    for i, (cosines, idx) in enumerate(zip(test_cosine, test_cosine_idx)):
        pred_ids = train_label[idx]
        for cosine, pred_id in zip(cosines, pred_ids):
            test_all_knn[i][pred_id] = max(test_all_knn[i][pred_id], cosine)
    return test_all_knn


def knn_both_feat(n_class, train_feat1, train_feat2, test_feat1, test_feat2, train_label):
    # knn with both feats for train
    train_feat12 = np.concatenate([train_feat1, train_feat2], axis=0)
    knn1_both_mat = knn_all_pred(n_class, len(train_label), test_feat1, train_feat12, train_label)
    knn2_both_mat = knn_all_pred(n_class, len(train_label), test_feat2, train_feat12, train_label)
    return (knn1_both_mat + knn2_both_mat) / 2


def binary_search_threshold(n_class: int, mat: torch.Tensor, new_ratio: float) -> float:
    ok, ng = 0.0, 1.0
    for _ in range(30):
        mid = (ok + ng) / 2
        out_new = torch.cat([mat, torch.full((mat.shape[0], 1), mid, device=mat.device)], dim=1)
        if (out_new.argmax(1) == n_class).to(float).mean() <= new_ratio:
            ok = mid
        else:
            ng = mid
    return ok


def make_submission(
    train_paths: List[str],
    test_paths: List[str],
    args: argparse.Namespace,
    cfg: Config,
    new_ratios: List[float],
    knn_ratio: float = 0.5,
):
    os.makedirs("submission", exist_ok=True)
    # ensemble
    csr_sum = None
    species_mats = []
    for train_path, test_path in zip(train_paths, test_paths):
        print(train_path, test_path)
        train_results, test_results = load_results(train_path), load_results(test_path)
        knn_csr = scipy.sparse.csr_matrix(
            knn_both_feat(
                cfg.num_classes,
                train_results["embed_features1"],
                train_results["embed_features2"],
                test_results["embed_features1"],
                test_results["embed_features2"],
                train_results["label"],
            )
        )
        logit_mat_csr = scipy.sparse.csr_matrix(
            restore_all_pred(cfg.num_classes, test_results["pred_logit"], test_results["pred_idx"])
        )
        mat_csr = knn_csr * knn_ratio + logit_mat_csr * (1 - knn_ratio)
        csr_sum = mat_csr if csr_sum is None else csr_sum + mat_csr
        species_mats.append(test_results["pred_species"])
    ensembled_mat = (csr_sum / len(train_paths)).todense()
    species_mat = np.stack(species_mats).mean(0)
    out = torch.tensor(ensembled_mat)

    # make submission from ensembled prediction
    for new_ratio in new_ratios:
        threshold = binary_search_threshold(cfg.num_classes, out, new_ratio)
        print(f"new_ratio: {new_ratio}, selected threshold: {threshold}")
        out_new = torch.cat([out, torch.full((out.shape[0], 1), threshold, device=out.device)], dim=1)
        top5 = out_new.topk(5)[1]

        # make csv
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.classes_ = np.load(f"{args.in_base_dir}/individual_id.npy", allow_pickle=True)
        assert cfg.num_classes == len(label_encoder.classes_)

        def make_str(id_list):
            return " ".join(
                "new_individual" if x == cfg.num_classes else label_encoder.inverse_transform([x])[0] for x in id_list
            )

        df = pd.read_csv(f"{args.in_base_dir}/sample_submission.csv")
        df["predictions"] = [make_str(id_list) for id_list in top5]
        df.to_csv(f"submission/{args.out_prefix}-{new_ratio}-{threshold}.csv", index=False, columns=["image", "predictions"])

    # generate pseudo label
    df = pd.read_csv(f"{args.in_base_dir}/sample_submission.csv")
    top1_conf, top1 = out.max(1)
    df["individual_id"] = label_encoder.inverse_transform(top1)
    df["conf"] = top1_conf
    le_species = preprocessing.LabelEncoder()
    le_species.classes_ = np.load(f"{args.in_base_dir}/species.npy", allow_pickle=True)
    df["species"] = le_species.inverse_transform(species_mat.argmax(1))
    df.to_csv(
        f"submission/pseudo_label_{args.out_prefix}.csv", index=False, columns=["image", "individual_id", "conf", "species"]
    )


if __name__ == "__main__":
    args = parse()
    cfg = load_config("config/default.yaml", "config/default.yaml")
    train_paths, test_paths = [], []
    for path in args.model_dirs:
        for bbox_name in cfg.test_bboxes:
            train_paths.append(f"{path}/train_{bbox_name}_results.npz")
            test_paths.append(f"{path}/test_{bbox_name}_results.npz")
    make_submission(train_paths, test_paths, args, cfg, [0.165], knn_ratio=0.5)
