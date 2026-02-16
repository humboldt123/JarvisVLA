"""
Describe mode: spawn a Minecraft environment and query the model in text/VLM mode.

Instead of playing the game (action tokens), this asks the model to describe
what it sees, answer knowledge questions, etc. Useful for testing the model's
visual understanding and language capabilities.

Usage:
    # Single question about current scene
    python jarvisvla/evaluate/describe.py \
        --base-url http://localhost:8000/v1 \
        --checkpoints CraftJarvis/JarvisVLA-Qwen2-VL-7B \
        --env-config kill/kill_zombie \
        --question "What do you see in front of you?"

    # Interactive mode: keep asking questions
    python jarvisvla/evaluate/describe.py \
        --base-url http://localhost:8000/v1 \
        --checkpoints CraftJarvis/JarvisVLA-Qwen2-VL-7B \
        --env-config mine/mine_wood \
        --interactive

    # Auto-describe: take actions AND describe every N steps
    python jarvisvla/evaluate/describe.py \
        --base-url http://localhost:8000/v1 \
        --checkpoints CraftJarvis/JarvisVLA-Qwen2-VL-7B \
        --env-config kill/kill_zombie \
        --auto-describe --describe-interval 20 --max-frames 200
"""

import argparse
import os
import json
from pathlib import Path
from datetime import datetime

import hydra
import numpy as np
from rich import console as rich_console
from PIL import Image

from minestudio.simulator import MinecraftSim
from minestudio.simulator.entry import CameraConfig
from minestudio.simulator.callbacks import (
    SpeedTestCallback,
    RecordCallback,
    FastResetCallback,
    InitInventoryCallback,
    TaskCallback,
    RewardsCallback,
    SummonMobsCallback,
    CommandsCallback,
)

from jarvisvla.evaluate import agent_wrapper

console = rich_console.Console()


def setup_env(env_config, record_path=None):
    """Set up a Minecraft environment from a config name like 'kill/kill_zombie'."""
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    config_path_obj = Path(f"{env_config}.yaml")
    config_name = config_path_obj.stem
    config_dir = os.path.join("./config", config_path_obj.parent)
    hydra.initialize(config_path=config_dir, version_base='1.3')
    cfg = hydra.compose(config_name=config_name)

    camera_cfg = CameraConfig(**cfg.camera_config)

    callbacks = [
        FastResetCallback(
            biomes=cfg.candidate_preferred_spawn_biome,
            random_tp_range=cfg.random_tp_range,
            start_time=cfg.start_time,
        ),
        SpeedTestCallback(50),
        TaskCallback(getattr(cfg, "task_conf", None)),
        RewardsCallback(getattr(cfg, "reward_conf", None)),
        InitInventoryCallback(
            init_inventory=getattr(cfg, "init_inventory", []),
            distraction_level=getattr(cfg, "inventory_distraction_level", [0]),
        ),
        CommandsCallback(getattr(cfg, "command", []),),
    ]

    if record_path:
        callbacks.append(RecordCallback(record_path=record_path, fps=30, show_actions=False))

    if cfg.mobs:
        callbacks.append(SummonMobsCallback(cfg.mobs))

    env = MinecraftSim(
        action_type="env",
        seed=cfg.seed,
        obs_size=cfg.origin_resolution,
        render_size=cfg.resize_resolution,
        camera_config=camera_cfg,
        preferred_spawn_biome=getattr(cfg, "preferred_spawn_biome", None),
        callbacks=callbacks,
    )
    return env, cfg


def save_screenshot(frame, path):
    """Save a numpy frame as a JPEG screenshot."""
    Image.fromarray(frame).save(path, format="JPEG", quality=90)


def make_folder_name(args):
    """Build folder name matching evaluate.py pattern: {model}-{task}"""
    model_ref = args.checkpoints.split('/')[-1]
    task_ref = args.env_config.split('/')[-1]
    return f"{model_ref}-{task_ref}-describe"


def describe_only(args):
    """Spawn environment, take a screenshot, ask questions."""
    console.log(f"Setting up environment: {args.env_config}")
    record_path = None
    if args.record:
        record_path = os.path.join(args.video_main_fold, make_folder_name(args))

    env, cfg = setup_env(args.env_config, record_path)
    obs, info = env.reset()

    # Let the env settle for a few frames
    for _ in range(5):
        action = env.noop_action()
        obs, _, _, _, info = env.step(action)

    console.log("Environment ready.")

    # Set up agent (we only use query_vlm, not forward)
    agent = agent_wrapper.VLLM_AGENT(
        checkpoint_path=args.checkpoints,
        base_url=args.base_url,
        history_num=0,
        action_chunk_len=1,
        instruction_type="normal",
        temperature=0.5,
    )

    frame = info["pov"]
    screenshot_dir = Path(args.video_main_fold) / make_folder_name(args) / "screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if args.interactive:
        # Interactive loop
        console.log("[bold green]Interactive mode. Type 'quit' to exit, 'look' to save screenshot.[/bold green]")
        step = 0
        while True:
            try:
                question = input("\nQuestion> ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if question.lower() in ("quit", "exit", "q"):
                break

            if question.lower() == "look":
                path = screenshot_dir / f"frame_{step:04d}.jpg"
                save_screenshot(frame, str(path))
                console.log(f"Screenshot saved to {path}")
                continue

            if question.lower().startswith("step"):
                # Take some actions to move around
                n = 10
                parts = question.split()
                if len(parts) > 1:
                    try:
                        n = int(parts[1])
                    except ValueError:
                        pass
                console.log(f"Taking {n} random steps...")
                for _ in range(n):
                    action = env.noop_action()
                    # Random camera movement to look around
                    action['camera'] = np.array([
                        np.random.uniform(-2, 2),
                        np.random.uniform(-2, 2),
                    ])
                    obs, _, _, _, info = env.step(action)
                    frame = info["pov"]
                    step += 1
                console.log("Done moving.")
                continue

            console.log("Querying model...")
            try:
                answer = agent.query_vlm(frame, question, include_image=True)
                console.log(f"[bold cyan]Answer:[/bold cyan] {answer}")
                results.append({"step": step, "question": question, "answer": answer})
            except Exception as e:
                console.log(f"[red]Error: {e}[/red]")

    else:
        # Single question mode
        questions = [args.question] if args.question else [
            "Describe what you see in this Minecraft scene.",
            "What biome are you in?",
            "What entities or mobs can you see?",
        ]

        for q in questions:
            console.log(f"\n[bold]Q: {q}[/bold]")
            try:
                answer = agent.query_vlm(frame, q, include_image=True)
                console.log(f"[cyan]A: {answer}[/cyan]")
                results.append({"question": q, "answer": answer})
            except Exception as e:
                console.log(f"[red]Error: {e}[/red]")

    # Save screenshot + results
    path = screenshot_dir / f"describe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    save_screenshot(frame, str(path))
    console.log(f"\nScreenshot saved: {path}")

    if results:
        results_path = screenshot_dir / f"describe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        console.log(f"Results saved: {results_path}")

    env.close()


def auto_describe(args):
    """Run the agent in action mode, but periodically describe the scene in text mode."""
    console.log(f"Setting up environment: {args.env_config}")
    video_fold = os.path.join(args.video_main_fold, make_folder_name(args))
    Path(video_fold).mkdir(parents=True, exist_ok=True)

    env, cfg = setup_env(args.env_config, record_path=video_fold)
    obs, info = env.reset()
    env.action_type = "agent"

    agent = agent_wrapper.VLLM_AGENT(
        checkpoint_path=args.checkpoints,
        base_url=args.base_url,
        history_num=args.history_num,
        action_chunk_len=1,
        instruction_type=args.instruction_type,
        temperature=args.temperature,
    )

    instructions = [item["text"] for item in cfg.task_conf]
    describe_questions = [
        "What do you see in front of you right now?",
        "Describe the current scene.",
        "What are you looking at?",
    ]

    results = []
    screenshot_dir = Path(video_fold) / "screenshots"
    screenshot_dir.mkdir(exist_ok=True)

    console.log(f"Running {args.max_frames} frames, describing every {args.describe_interval} steps...")
    console.log(f"Task: {instructions}")

    for step in range(args.max_frames):
        frame = info["pov"]

        # Take an action (normal gameplay)
        action = agent.forward(
            [frame], instructions,
            verbos=False, need_crafting_table=False,
        )
        obs, reward, terminated, truncated, info = env.step(action)

        if reward > 0:
            console.log(f"[bold green]Step {step}: Task succeeded! reward={reward}[/bold green]")

        # Periodically describe
        if step > 0 and step % args.describe_interval == 0:
            q = describe_questions[step // args.describe_interval % len(describe_questions)]
            try:
                answer = agent.query_vlm(frame, q, include_image=True)
                console.log(f"\n[bold]Step {step} - Q: {q}[/bold]")
                console.log(f"[cyan]A: {answer}[/cyan]\n")
                results.append({"step": step, "question": q, "answer": answer, "reward": float(reward)})

                # Save screenshot at this step
                path = screenshot_dir / f"step_{step:04d}.jpg"
                save_screenshot(frame, str(path))
            except Exception as e:
                console.log(f"[red]Step {step} describe error: {e}[/red]")

    # Save all results
    results_path = Path(video_fold) / "descriptions.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    console.log(f"\nDescriptions saved: {results_path}")
    console.log(f"Video/screenshots in: {video_fold}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Describe mode for JarvisVLA")
    parser.add_argument('--base-url', type=str, default='http://localhost:8000/v1')
    parser.add_argument('--checkpoints', type=str, default='CraftJarvis/JarvisVLA-Qwen2-VL-7B')
    parser.add_argument('--env-config', type=str, default='kill/kill_zombie')
    parser.add_argument('--video-main-fold', type=str, default='logs/')
    parser.add_argument('--record', action='store_true', help='Record video')

    # Single question mode
    parser.add_argument('--question', '-q', type=str, default=None,
                        help='Single question to ask about the scene')

    # Interactive mode
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Interactive Q&A mode')

    # Auto-describe mode (play + describe)
    parser.add_argument('--auto-describe', action='store_true',
                        help='Play the game and periodically describe the scene')
    parser.add_argument('--describe-interval', type=int, default=20,
                        help='Describe every N steps (for auto-describe mode)')
    parser.add_argument('--max-frames', type=int, default=200)
    parser.add_argument('--temperature', '-t', type=float, default=0.7)
    parser.add_argument('--history-num', type=int, default=2)
    parser.add_argument('--instruction-type', type=str, default='normal')

    args = parser.parse_args()

    if args.auto_describe:
        auto_describe(args)
    else:
        describe_only(args)
