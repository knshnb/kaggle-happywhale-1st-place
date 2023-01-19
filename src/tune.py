import argparse
import copy

import optuna

from config.config import load_config
from src.dataset import load_df
from src.train import train


def parse():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for HappyWhale")
    parser.add_argument("--out_base_dir", default="result")
    parser.add_argument("--in_base_dir", default="input")
    parser.add_argument("--exp_name", default="tune_tmp")
    parser.add_argument("--load_snapshot", action="store_true")
    parser.add_argument("--save_checkpoint", action="store_true")
    parser.add_argument("--wandb_logger", action="store_true")
    parser.add_argument("--config_path", default="config/debug.yaml")
    parser.add_argument("--rdb_url", default="sqlite:///tmp.db")
    return parser.parse_args()


def main():
    base_args = parse()
    base_cfg = load_config(base_args.config_path, "config/default.yaml")
    df = load_df(base_args.in_base_dir, base_cfg, "train.csv", True)

    def objective(trial: optuna.trial.Trial) -> float:
        args = copy.deepcopy(base_args)
        args.exp_name = f"{args.exp_name}/{trial.number}"

        cfg = copy.deepcopy(base_cfg)
        # dynamic arcface parameters
        cfg["s_id"] = trial.suggest_float("s_id", 10.0, 80.0)
        cfg["s_species"] = trial.suggest_float("s_species", 10.0, 80.0)
        cfg["loss_id_ratio"] = trial.suggest_float("loss_id_ratio", 0.2, 1.0)
        cfg["margin_power_id"] = trial.suggest_float("margin_power_id", -0.8, -0.05)
        cfg["margin_power_species"] = trial.suggest_float("margin_power_species", -0.8, -0.05)
        cfg["margin_coef_id"] = trial.suggest_float("margin_coef_id", 0.2, 1.0)
        cfg["margin_coef_species"] = trial.suggest_float("margin_coef_species", 0.2, 1.0)

        score = train(df, args, cfg, 0, optuna_trial=trial)
        assert score is not None
        return score

    storage = optuna.storages.RDBStorage(
        url=base_args.rdb_url,
        heartbeat_interval=60,
        grace_period=120,
        failed_trial_callback=optuna.storages.RetryFailedTrialCallback(),
    )
    study = optuna.create_study(
        direction="maximize",
        storage=storage,
        study_name=base_args.exp_name,
        load_if_exists=True,
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(objective, callbacks=[optuna.study.MaxTrialsCallback(500)])


if __name__ == "__main__":
    main()
