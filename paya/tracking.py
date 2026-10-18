"""Optional W&B tracking with an explicit credential and configuration allowlist."""

import json
import os
import uuid
from pathlib import Path

from dotenv import dotenv_values


class Tracker:
    """Provide optional W&B lifecycle hooks without exposing arbitrary configuration."""

    def __init__(self, directory, config, resume_id=None):
        """Start or resume tracking; disabled mode does not read credentials or contact W&B."""
        self.run = None
        if config.get("wandb", {}).get("mode", "disabled") == "disabled":
            return
        for path in (Path.cwd().parent / ".env", Path.cwd() / ".env"):
            if path.exists():
                values = dotenv_values(path, interpolate=False)
                for key in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_PROJECT"):
                    value = values.get(key) or (
                        values.get("WANDB") if key == "WANDB_API_KEY" else None
                    )
                    if value:
                        os.environ.setdefault(key, value)
        import wandb

        self.run = wandb.init(
            id=resume_id or uuid.uuid4().hex[:12],
            resume="must" if resume_id else "never",
            project=config.get("wandb", {}).get("project", "paya"),
            mode=config["wandb"]["mode"],
            name=Path(directory).name,
            dir=str(directory),
            config={k: config[k] for k in ("head", "optim", "seed")},
            settings=wandb.Settings(
                console="off",
                disable_git=True,
                disable_code=True,
                save_code=False,
                x_disable_meta=True,
                x_save_requirements=False,
                quiet=True,
            ),
        )
        (Path(directory) / "wandb.json").write_text(
            json.dumps({"id": self.run.id, "url": self.run.url})
        )

    @property
    def id(self):
        """Return the active tracking ID, or None when tracking is disabled."""
        return self.run.id if self.run else None

    def log(self, values, step):
        """Log metrics at an optimizer step when tracking is enabled."""
        if self.run:
            self.run.log({"optimizer_step": step, **values})

    def finish(self):
        """Close an enabled tracking run; otherwise do nothing."""
        if self.run:
            self.run.finish()
