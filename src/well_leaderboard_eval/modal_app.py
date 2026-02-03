import os
import modal

APP_NAME = "the-well-leaderboard-eval-v110"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("fno-well-datasets", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "well-leaderboard-eval",  # if you publish to pip; for editable, mount repo instead
        "modal",
    )
)

# If you are developing (not published), prefer mounting your repo and pip install -e .
# Example:
# image = modal.Image.debian_slim(python_version="3.11").apt_install("git").pip_install("modal").run_commands(
#   "pip install -e /repo"
# )

@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={VOLUME_PATH: volume},
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret-2")],
)
def run_eval():
    from well_leaderboard_eval import EvalConfig, evaluate_dataset

    cfg = EvalConfig(
        dataset_name=os.environ.get("DATASET_NAME", "helmholtz_staircase"),
        split_name=os.environ.get("SPLIT_NAME", "test"),
        model_id=os.environ.get("MODEL_ID", None),
        n_steps_input=int(os.environ.get("N_STEPS_INPUT", "4")),
        rollout=int(os.environ.get("ROLLOUT", "30")),
        seed=int(os.environ.get("SEED", "0")),
        base_path=VOLUME_PATH,
        out_dir=VOLUME_PATH,                 # write JSON into the volume
        download_if_missing=True,
        device="cuda",
        deterministic=True,
    )

    results = evaluate_dataset(cfg)
    # Persist volume changes (datasets + json)
    volume.commit()
    return results


@app.local_entrypoint()
def main():
    print(run_eval.remote())
