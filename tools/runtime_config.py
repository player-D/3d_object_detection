import glob
import os


def _normalize_path(path_value):
    if not path_value:
        return ""
    return os.path.abspath(os.path.expanduser(path_value))


def _collect_checkpoint_candidates(saved_models_dir="saved_models"):
    pattern = os.path.join(saved_models_dir, "**", "best_model.pth")
    return [
        os.path.abspath(path)
        for path in glob.glob(pattern, recursive=True)
        if os.path.isfile(path)
    ]


def find_latest_checkpoint(saved_models_dir="saved_models"):
    candidates = _collect_checkpoint_candidates(saved_models_dir)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def resolve_checkpoint_path(checkpoint_path=None, env_var="TDR_CHECKPOINT"):
    explicit = checkpoint_path or os.environ.get(env_var, "").strip()
    if explicit:
        resolved = _normalize_path(explicit)
        if os.path.isfile(resolved):
            return resolved
        raise FileNotFoundError(f"Checkpoint not found: {resolved}")

    checkpoint_root = _normalize_path(os.environ.get("TDR_CHECKPOINT_ROOT", "").strip()) or "saved_models"
    latest = find_latest_checkpoint(checkpoint_root)
    if latest:
        return latest

    legacy_default = _normalize_path("./saved_models/04_10_10-21/best_model.pth")
    if os.path.isfile(legacy_default):
        return legacy_default

    raise FileNotFoundError(
        "No checkpoint was found. Set TDR_CHECKPOINT or place best_model.pth under saved_models/<run_id>/."
    )


def resolve_sample_indices_path(checkpoint_path, sample_indices_path=None, env_var="TDR_SAMPLE_INDICES"):
    explicit = sample_indices_path or os.environ.get(env_var, "").strip()
    if explicit:
        resolved = _normalize_path(explicit)
        if os.path.isfile(resolved):
            return resolved
        raise FileNotFoundError(f"Sample indices file not found: {resolved}")

    if checkpoint_path:
        candidate = os.path.join(os.path.dirname(checkpoint_path), "sample_indices.json")
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)

    return None


def resolve_root_dir(path_value=None, default_root="output", env_var=None):
    configured = ""
    if env_var:
        configured = os.environ.get(env_var, "").strip()
    root = path_value or configured or default_root
    resolved = _normalize_path(root)
    os.makedirs(resolved, exist_ok=True)
    return resolved


def create_training_run_dirs(checkpoint_root=None, log_root=None, run_name=None):
    import datetime

    run_id = run_name or datetime.datetime.now().strftime("%m_%d_%H-%M")
    checkpoint_root = resolve_root_dir(checkpoint_root, default_root="saved_models", env_var="TDR_CHECKPOINT_ROOT")
    log_root = resolve_root_dir(log_root, default_root="logs", env_var="TDR_LOG_ROOT")

    save_dir = os.path.join(checkpoint_root, run_id)
    log_dir = os.path.join(log_root, run_id)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    return {
        "run_id": run_id,
        "save_dir": save_dir,
        "log_dir": log_dir,
        "checkpoint_root": checkpoint_root,
        "log_root": log_root,
    }


def resolve_output_root(default_root="output", env_var="TDR_OUTPUT_ROOT"):
    return resolve_root_dir(default_root=default_root, env_var=env_var)
