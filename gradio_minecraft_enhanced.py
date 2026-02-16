#!/usr/bin/env python3
"""
Enhanced Gradio interface for JarvisVLA with chat, video recording, and real-time control
"""

import gradio as gr
import numpy as np
from pathlib import Path
import queue
from PIL import Image
import hydra
from rich import console
from datetime import datetime
import os
import json
import re
import sys
import logging
import traceback as tb_module

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import (
    SpeedTestCallback,
    RecordCallback,
    FastResetCallback,
    InitInventoryCallback,
    TaskCallback,
    RewardsCallback,
)
from minestudio.simulator.entry import CameraConfig

from jarvisvla.evaluate import agent_wrapper

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/vvm33/JarvisVLA/gradio.log', mode='a'),
        logging.StreamHandler(sys.stdout)
    ]
)
for lib in ["httpcore", "httpx", "urllib3", "openai"]:
    logging.getLogger(lib).setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

# Load special tokens to filter out of display
SPECIAL_TOKENS_PATH = Path(__file__).parent / "assets" / "special_token.json"
SPECIAL_TOKENS = []
try:
    with open(SPECIAL_TOKENS_PATH, 'r') as f:
        SPECIAL_TOKENS = json.load(f)
    logger.info(f"Loaded {len(SPECIAL_TOKENS)} special tokens")
except Exception as e:
    logger.warning(f"Could not load special tokens: {e}")


def clean_llm_output(text):
    """Remove action tokens, chat template tokens, and degenerate repetition.
    Returns (cleaned_text, has_meaningful_content) tuple."""
    if not text:
        return "", False

    cleaned = text

    # Remove action-related special tokens
    for token in SPECIAL_TOKENS:
        if 'reserved_special_token' in token or 'visual_' in token or 'point_start' in token or 'point_end' in token:
            cleaned = cleaned.replace(token, "")

    # Strip vLLM list wrappers and Python-repr artifacts
    cleaned = re.sub(r"^\s*\[\s*'", "", cleaned)
    cleaned = re.sub(r"'\s*\]\s*$", "", cleaned)
    cleaned = re.sub(r"\['\s*", "", cleaned)   # stray ['
    cleaned = re.sub(r"\s*'\]", "", cleaned)    # stray ']
    cleaned = re.sub(r"\('\s*'[^']*'\s*\)", "", cleaned)  # ('...')
    cleaned = re.sub(r"\[''?\]", "", cleaned)   # [''] or []

    # Strip chat template and structural tokens
    for tag in ['<|im_start|>', '<|im_end|>', '<|endoftext|>',
                '<think>', '</think>', '<grounding>', '</grounding>',
                '<motion>', '</motion>']:
        cleaned = cleaned.replace(tag, '')

    # Strip degenerate repetition
    cleaned = re.sub(r'(\b\w+\b)(\s*\1){2,}', r'\1', cleaned)

    # Clean up whitespace, stray punctuation, and role labels
    cleaned = re.sub(r"[',()]+\s*[',()]+", " ", cleaned)  # runs of quotes/parens
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned)
    cleaned = cleaned.strip().strip("',()[] \n")
    cleaned = re.sub(r'^(user|assistant|system)\s*', '', cleaned, flags=re.IGNORECASE).strip()

    # Meaningful = more than 3 chars of actual alphanumeric text
    content_check = re.sub(r'[^a-zA-Z0-9 ]', '', cleaned).strip()
    has_meaningful = len(content_check) > 5

    return cleaned if cleaned else "", has_meaningful


def discover_configs():
    """Discover all available task configs"""
    config_dir = Path("./jarvisvla/evaluate/config")
    configs = {}
    for category_dir in config_dir.iterdir():
        if category_dir.is_dir():
            category = category_dir.name
            for config_file in category_dir.glob("*.yaml"):
                if config_file.stem != "base":
                    config_path = f"{category}/{config_file.stem}"
                    display_name = f"{category.upper()}: {config_file.stem.replace('_', ' ').title()}"
                    configs[display_name] = config_path
    return configs


class MinecraftGradioInterface:
    def __init__(self, checkpoint_path, base_url="http://localhost:8000/v1"):
        self.checkpoint_path = checkpoint_path
        self.base_url = base_url
        self.env = None
        self.agent = None
        self.running = False
        self.console = console.Console()
        self.current_instruction = None
        self.instruction_queue = queue.Queue()
        self.vlm_query_queue = queue.Queue()
        self.video_dir = Path("./videos")
        self.video_dir.mkdir(exist_ok=True)
        self.current_frame = None

    def initialize_env(self, env_config="mine/mine_wood", record_video=True):
        """Initialize Minecraft environment"""
        hydra.core.global_hydra.GlobalHydra.instance().clear()

        if "/" in env_config:
            config_dir = f"./jarvisvla/evaluate/config/{env_config.split('/')[0]}"
            config_name = env_config.split('/')[1]
        else:
            config_dir = "./jarvisvla/evaluate/config/mine"
            config_name = "mine_wood"

        try:
            hydra.initialize(config_path=config_dir, version_base='1.3')
            cfg = hydra.compose(config_name=config_name)
        except Exception as e:
            logger.warning(f"Could not load config, using defaults: {e}")
            cfg = type('Config', (), {
                'init_inventory': [],
                'candidate_preferred_spawn_biome': ['plains', 'forest'],
                'inventory_distraction_level': [0],
            })()

        # ORDER MATTERS: RecordCallback before InitInventoryCallback
        # (InitInventoryCallback.after_reset calls raw env.step → obs has 'pov' not 'image')
        callbacks = [
            FastResetCallback(
                biomes=getattr(cfg, 'candidate_preferred_spawn_biome', ['plains', 'forest']),
                random_tp_range=getattr(cfg, 'random_tp_range', 5000),
                start_time=getattr(cfg, 'start_time', 0),
            ),
            SpeedTestCallback(50),
        ]

        if record_video:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_dir = self.video_dir / f"session_{config_name}_{timestamp}"
            callbacks.append(RecordCallback(record_path=str(video_dir), fps=20, frame_type="obs"))

        init_inventory = getattr(cfg, 'init_inventory', [])
        if init_inventory:
            callbacks.append(InitInventoryCallback(
                init_inventory=init_inventory,
                distraction_level=getattr(cfg, 'inventory_distraction_level', [0])
            ))

        if hasattr(cfg, 'task_conf') and cfg.task_conf:
            callbacks.append(TaskCallback(cfg.task_conf))
        if hasattr(cfg, 'reward_conf') and cfg.reward_conf:
            callbacks.append(RewardsCallback(cfg.reward_conf))

        camera_config = CameraConfig(
            camera_maxval=10, camera_binsize=1,
            camera_quantization_scheme="mu_law", camera_mu=20,
        )

        self.env = MinecraftSim(
            action_type="agent",
            seed=np.random.randint(0, 1000000),
            obs_size=(640, 360), render_size=(640, 360),
            preferred_spawn_biome='plains',
            callbacks=callbacks, camera_config=camera_config,
        )
        return self.env.reset()

    def initialize_agent(self, instruction_type='normal'):
        """Initialize the agent"""
        self.agent = agent_wrapper.VLLM_AGENT(
            checkpoint_path=self.checkpoint_path,
            base_url=self.base_url,
            history_num=0,
            action_chunk_len=1,
            instruction_type=instruction_type,
            temperature=0.5,
        )

    def _extract_frame(self, obs):
        """Extract numpy frame from obs dict, handling both wrapped and raw formats."""
        if isinstance(obs, dict):
            frame = obs.get('image')
            if frame is None:
                frame = obs.get('pov')
            if frame is None:
                frame = next((v for v in obs.values() if isinstance(v, np.ndarray) and len(v.shape) == 3), None)
        else:
            frame = obs
        return frame

    def run_inference(self, instruction, env_config="mine/mine_wood",
                     run_indefinitely=False, record_video=False,
                     auto_describe=True, instruction_type='normal'):
        """Main game loop. Yields (frame, status, steps, thinking, chat) to Gradio."""
        self.running = True
        self.current_instruction = instruction
        max_steps = 999999 if run_indefinitely else 600
        chat_history = []
        jpeg_path = Path("/tmp/jarvis_frame.jpg")

        try:
            # --- Init ---
            yield None, "Initializing environment...", 0, "", chat_history

            obs, info = self.initialize_env(env_config, record_video)
            logger.info(f"Environment ready, obs keys: {list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")

            yield None, f"Loading agent ({instruction_type})...", 0, "", chat_history
            self.initialize_agent(instruction_type=instruction_type)

            chat_history.append({"role": "assistant", "content": f"Starting task: {instruction}"})
            yield None, f"Running: {instruction}", 0, "", chat_history

            # --- Game loop ---
            step = 0
            success = False

            while step < max_steps and self.running:
                # Check for new task instruction
                try:
                    new_instruction = self.instruction_queue.get_nowait()
                    self.current_instruction = new_instruction
                    chat_history.append({"role": "user", "content": new_instruction})
                    chat_history.append({"role": "assistant", "content": f"Switching to: {new_instruction}"})
                except queue.Empty:
                    pass

                # Check for VLM questions
                vlm_response = ""
                try:
                    vlm_question = self.vlm_query_queue.get_nowait()
                    if vlm_question and self.agent and self.current_frame is not None:
                        chat_history.append({"role": "user", "content": vlm_question})
                        try:
                            vlm_answer = self.agent.query_vlm(self.current_frame, vlm_question)
                            logger.info(f"VLM answer: {repr(vlm_answer[:200] if vlm_answer else None)}")
                            vlm_cleaned, _ = clean_llm_output(vlm_answer)
                            vlm_response = vlm_cleaned if vlm_cleaned else "(model produced no text)"
                        except Exception as e:
                            vlm_response = f"Error: {e}"
                            logger.error(f"VLM query error: {e}", exc_info=True)
                        chat_history.append({"role": "assistant", "content": vlm_response})
                except queue.Empty:
                    pass

                # Extract frame
                frame = self._extract_frame(obs)
                if frame is None or not isinstance(frame, np.ndarray):
                    logger.error(f"Step {step}: bad frame, obs keys={list(obs.keys()) if isinstance(obs, dict) else 'N/A'}")
                    break

                self.current_frame = frame
                Image.fromarray(frame).save(jpeg_path, format="JPEG", quality=80)

                # Periodic status update in chat
                if auto_describe and step > 0 and step % 200 == 0:
                    chat_history.append({"role": "assistant", "content": f"[Step {step}] Still working on: {self.current_instruction}"})

                # Get action
                try:
                    action = self.agent.forward(
                        observations=[frame],
                        instructions=[self.current_instruction],
                        verbos=False,
                        need_crafting_table=False,
                    )
                    action_str = f"Buttons: {action.get('buttons', 0)}, Camera: {action.get('camera', 0)}"
                except Exception as e:
                    logger.error(f"agent.forward error: {e}")
                    action = {'buttons': 0, 'camera': 220}
                    action_str = f"NULL (error: {e})"

                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(action)
                step += 1

                if reward > 0:
                    success = True

                # Build displays
                status = f"Step {step}" + (f"/{max_steps}" if not run_indefinitely else "")
                if success:
                    status += f" | Reward: {reward:.2f}"

                thinking = f"**Task:** {self.current_instruction}\n\n**Action:** {action_str}\n\n"
                if vlm_response:
                    thinking += f"**Agent:** {vlm_response}\n\n"
                thinking += f"**Step:** {step} | **Reward:** {reward}\n"

                # Yield every 5 steps, or immediately on VLM response / success
                if step <= 1 or step % 5 == 0 or success or vlm_response:
                    yield str(jpeg_path), status, step if run_indefinitely else min(step, max_steps), thinking, chat_history

                if success and step > 50 and not run_indefinitely:
                    chat_history.append({"role": "assistant", "content": f"Task completed in {step} steps!"})
                    yield str(jpeg_path), "Task completed!", step, thinking, chat_history
                    break

            if not success and not run_indefinitely:
                chat_history.append({"role": "assistant", "content": f"Stopped after {step} steps"})
                yield str(jpeg_path), "Stopped", step, thinking, chat_history

        except Exception as e:
            full_tb = tb_module.format_exc()
            logger.error(f"EXCEPTION: {e}\n{full_tb}")
            yield None, f"Error: {e}", 0, f"```\n{full_tb}\n```", chat_history

        finally:
            if self.env:
                self.env.close()
                self.env = None
            self.running = False

    def stop(self):
        self.running = False
        if self.env:
            self.env.close()
            self.env = None

    def send_instruction(self, instruction):
        if instruction.strip():
            self.instruction_queue.put(instruction.strip())
            return "", f"Sent: {instruction}"
        return "", ""

    def ask_agent(self, question):
        if question.strip():
            self.vlm_query_queue.put(question.strip())
            return "", f"Queued: {question}"
        return "", ""


def create_gradio_interface(checkpoint_path, base_url="http://localhost:8000/v1"):
    interface = MinecraftGradioInterface(checkpoint_path, base_url)
    available_configs = discover_configs()

    with gr.Blocks(title="JarvisVLA") as demo:
        gr.Markdown("# JarvisVLA - Minecraft Agent")

        with gr.Row():
            with gr.Column(scale=2):
                video_output = gr.Image(label="Gameplay", height=400)
                status_output = gr.Textbox(label="Status", lines=1)
                progress = gr.Number(label="Steps", value=0, interactive=False)
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop", variant="stop", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### Task Setup")
                initial_instruction = gr.Textbox(
                    label="Task Instruction", placeholder="e.g., mine wood", value="mine wood"
                )
                env_config_dropdown = gr.Dropdown(
                    choices=[(label, value) for label, value in zip(available_configs.keys(), available_configs.values())],
                    label="Environment", value="mine/mine_wood",
                )
                instruction_type_dropdown = gr.Dropdown(
                    choices=[
                        ("Normal - natural language instruction", "normal"),
                        ("Recipe - includes crafting grid (for craft/smelt)", "recipe"),
                        ("Simple - short thought prompt", "simple"),
                    ],
                    label="Prompt Mode", value="normal",
                    info="How the task instruction is formatted for the model. 'Recipe' is best for crafting tasks."
                )
                with gr.Row():
                    run_indefinitely = gr.Checkbox(label="Run Indefinitely", value=False)
                    record_video = gr.Checkbox(label="Record Video", value=False)
                    auto_describe = gr.Checkbox(label="Status Updates in Chat", value=True)

                gr.Markdown("### Change Task (while running)")
                chat_instruction = gr.Textbox(label="New Task", placeholder="e.g., find a village")
                send_btn = gr.Button("Send", size="sm")
                instruction_status = gr.Textbox(label="", lines=1)

                gr.Markdown("### Ask Agent (VLM Mode)")
                gr.Markdown("*Ask Minecraft knowledge questions. Best for recipes, crafting, game mechanics. Visual questions (\"what do you see\") have limited quality due to action fine-tuning.*")
                vlm_question_input = gr.Textbox(label="Question", placeholder="e.g., What do you see? How do you craft a pickaxe?")
                ask_btn = gr.Button("Ask", size="sm")
                ask_status = gr.Textbox(label="", lines=1)

                gr.Markdown("### Agent Status")
                thinking_output = gr.Markdown("Waiting to start...")

                gr.Markdown("### Chat")
                chat_output = gr.Chatbot(label="Chat", height=300)

        gr.Examples(
            examples=[
                ["mine wood", "mine/mine_wood", "normal", False, False, True],
                ["mine stone", "mine/mine_stone", "normal", False, False, True],
                ["craft a crafting table", "craft/craft_crafting_table", "recipe", False, False, True],
                ["kill zombie", "kill/kill_zombie", "normal", True, False, True],
            ],
            inputs=[initial_instruction, env_config_dropdown, instruction_type_dropdown,
                    run_indefinitely, record_video, auto_describe],
        )

        start_btn.click(
            fn=interface.run_inference,
            inputs=[initial_instruction, env_config_dropdown, run_indefinitely,
                    record_video, auto_describe, instruction_type_dropdown],
            outputs=[video_output, status_output, progress, thinking_output, chat_output],
        )
        stop_btn.click(fn=interface.stop)
        send_btn.click(fn=interface.send_instruction, inputs=[chat_instruction], outputs=[chat_instruction, instruction_status])
        ask_btn.click(fn=interface.ask_agent, inputs=[vlm_question_input], outputs=[vlm_question_input, ask_status])

    return demo


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="CraftJarvis/JarvisVLA-Qwen2-VL-7B")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_gradio_interface(args.checkpoint, args.base_url)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
