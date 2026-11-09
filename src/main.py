import argparse
from copy import copy
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value):
    path = Path(value).expanduser()
    return str((PROJECT_ROOT / path).resolve()) if not path.is_absolute() else str(path)


def get_args(argv=None):
    parser = argparse.ArgumentParser(description="Run CAAR retrieval-augmented generation.")
    parser.add_argument("-c", "--config_path", required=True, help="Path to a CAAR JSON configuration.")
    parser.add_argument("--model_name_or_path", help="Override the model ID or local checkpoint path.")
    parser.add_argument("--data_path", help="Override the dataset directory.")
    parser.add_argument("--output_dir", help="Override the output directory.")
    parser.add_argument("--sample", type=int, help="Number of examples; -1 means the entire dataset.")
    cli = parser.parse_args(argv)
    config_path = Path(cli.config_path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as fin:
        config = json.load(fin)
    for key in ("model_name_or_path", "data_path", "output_dir", "sample"):
        value = getattr(cli, key)
        if value is not None:
            config[key] = value
    config.setdefault("method", "caar")
    config.setdefault("shuffle", False)
    config.setdefault("use_counter", True)
    config.setdefault("sample", -1)
    config.setdefault("caar_layer_id", 20)
    if config["method"] != "caar":
        parser.error('This project supports method="caar". Set method to "caar" in your configuration.')
    if "llm_check_layer_id" in config:
        parser.error('Rename llm_check_layer_id to caar_layer_id.')
    required = ("model_name_or_path", "dataset", "data_path", "output_dir", "fewshot",
                "retriever", "retrieve_topk", "generate_max_length", "hallucination_threshold",
                "query_formulation")
    missing = [key for key in required if key not in config]
    if missing:
        parser.error("Missing configuration fields: " + ", ".join(missing))
    if config["sample"] < -1 or config["sample"] == 0:
        parser.error("--sample must be positive or -1.")
    if config["hallucination_threshold"] < 0:
        parser.error("CAAR uses negative-log scores and non-negative thresholds; "
                     "negate the old threshold when migrating.")
    if not 0 < config.get("attention_head_ratio", 0.25) <= 1:
        parser.error("attention_head_ratio must be in (0, 1].")
    if config["query_formulation"] in ("real_words", "real_words_head_select"):
        if not any(key in config for key in ("retrieve_keep_top_k", "retrieve_keep_ratio")):
            parser.error("Word selection requires retrieve_keep_top_k or retrieve_keep_ratio.")
    for key in ("data_path", "output_dir", "sgpt_encode_file_path", "passage_file"):
        if key in config:
            config[key] = project_path(config[key])
    for key in ("model_name_or_path", "sgpt_model_name_or_path"):
        if key in config:
            value = config[key]
            if value.startswith(("/", ".", "~")) or (PROJECT_ROOT / value).exists():
                config[key] = project_path(value)
    config["config_path"] = str(config_path)
    return argparse.Namespace(**config)


def create_run_dir(output_dir):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    for index in range(10000):
        run_dir = root / str(index)
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir
    raise RuntimeError(f"No free run directory under {root}")


def main():
    args = get_args()
    from tqdm import tqdm
    from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC
    from caar import CAARRAG

    dataset_types = {
        "strategyqa": StrategyQA, "2wikimultihopqa": WikiMultiHopQA,
        "hotpotqa": HotpotQA, "iirc": IIRC,
    }
    if args.dataset not in dataset_types:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    data = dataset_types[args.dataset](args.data_path)
    data.format(fewshot=args.fewshot)
    data = data.dataset
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        data = data.select(range(min(len(data), args.sample)))

    model = CAARRAG(args)
    run_dir = create_run_dir(args.output_dir)
    args.output_dir = str(run_dir)
    with (run_dir / "config.json").open("w", encoding="utf-8") as fout:
        json.dump(vars(args), fout, indent=4)
    logger.info("Output directory: %s", run_dir)
    with (run_dir / "output.txt").open("w", encoding="utf-8") as predictions_file:
        for index in tqdm(range(len(data))):
            last_counter = copy(model.counter)
            batch = data[index]
            pred = model.inference(
                batch["question"], batch["demo"], batch["case"],
            )
            result = {"qid": batch["qid"], "prediction": pred.strip()}
            if args.use_counter:
                result.update(model.counter.calc(last_counter))
            predictions_file.write(json.dumps(result) + "\n")
            predictions_file.flush()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
