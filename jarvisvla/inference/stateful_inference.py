"""
Stateful Inference for JarvisVLA

This module provides inference capabilities for the stateful VLA model with recurrent memory.

Key Features:
1. Maintains a memory state (512-dim) across timesteps
2. Uses W_in to project memory before forward pass
3. Appends memory token after visual/text tokens
4. Uses W_out to extract updated memory after forward pass
5. No auxiliary heads during inference (zero overhead)

Memory Flow at Inference:
    memory_t (512-dim)
        ↓ W_in
    projected_memory (3584-dim)
        ↓ append to [visual, text]
    forward through model
        ↓ extract last token hidden state
    raw_memory_hidden (3584-dim)
        ↓ W_out
    memory_{t+1} (512-dim) → carry to next timestep

Usage:
    inference = StatefulVLAInference.from_pretrained(checkpoint_path)
    inference.reset_state()
    
    for frame in video_stream:
        response = inference.generate(frame, text_prompt)
        # Memory automatically carries forward
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
import json

from transformers import Qwen2VLProcessor, Qwen2VLForConditionalGeneration

from jarvisvla.models.stateful_vla import StatefulJarvisVLA
from jarvisvla.train.auxiliary_heads import MemoryProjections


class StatefulVLAInference:
    """
    Wrapper for stateful VLA inference.
    
    Manages:
    - Model loading from checkpoints
    - Memory state initialization and propagation
    - Action/text generation with recurrent memory
    
    The auxiliary heads are NOT loaded during inference.
    """
    
    def __init__(
        self,
        model: StatefulJarvisVLA,
        processor: Qwen2VLProcessor,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.processor = processor
        self.device = device
        
        # Set to eval mode (disables auxiliary heads)
        self.model.eval()
        
        # Current memory state
        self.current_memory: Optional[torch.Tensor] = None
        self.batch_size: int = 0
    
    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str,
        base_model_name: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "StatefulVLAInference":
        """
        Load a stateful model from checkpoint.
        
        Checkpoint should contain:
        - pytorch_model.bin or model.safetensors (base model)
        - memory_projections.pt (W_in and W_out weights)
        - initial_memory.pt (learned initial memory)
        - stateful_config.json (configuration)
        
        Args:
            checkpoint_path: Path to checkpoint directory
            base_model_name: Name of base model (if not in config)
            device: Device to load model on
        
        Returns:
            StatefulVLAInference instance
        """
        checkpoint_path = Path(checkpoint_path)
        
        # Load config
        config_path = checkpoint_path / "stateful_config.json"
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            base_model_name = base_model_name or config.get('base_model_name')
            memory_dim = config.get('memory_dim', 512)
            hidden_dim = config.get('hidden_dim', 3584)
        else:
            memory_dim = 512
            hidden_dim = 3584
        
        if base_model_name is None:
            raise ValueError("Must provide base_model_name or have stateful_config.json")
        
        print(f"Loading base model: {base_model_name}")
        print(f"  Memory dim: {memory_dim}")
        print(f"  Hidden dim: {hidden_dim}")
        
        # Load processor
        processor = Qwen2VLProcessor.from_pretrained(
            base_model_name,
            do_rescale=False,
            patch_size=14,
            vision_feature_select_strategy="default",
        )
        
        # Add special tokens if needed
        special_tokens_path = checkpoint_path / "special_tokens.json"
        if special_tokens_path.exists():
            with open(special_tokens_path) as f:
                special_tokens = json.load(f)
            processor.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        # Load base model
        base_model = Qwen2VLForConditionalGeneration.from_pretrained(
            checkpoint_path,  # Load from checkpoint (has merged weights)
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.device_count() > 1 else None,
        )
        
        if torch.cuda.device_count() == 1:
            base_model = base_model.to(device)
        
        # Create stateful wrapper (without inventory head for inference)
        from jarvisvla.models.stateful_vla import StatefulJarvisVLA
        
        model = StatefulJarvisVLA(
            base_model=base_model,
            memory_dim=memory_dim,
            hidden_dim=hidden_dim,
            use_mlp_for_W_in=False,
        )
        
        # Load memory projections if saved separately
        proj_path = checkpoint_path / "memory_projections.pt"
        if proj_path.exists():
            model.memory_projections.load_state_dict(torch.load(proj_path, map_location=device))
            print(f"  Loaded memory projections from {proj_path}")
        
        # Load initial memory if saved separately
        init_mem_path = checkpoint_path / "initial_memory.pt"
        if init_mem_path.exists():
            model.initial_memory.data = torch.load(init_mem_path, map_location=device)
            print(f"  Loaded initial memory from {init_mem_path}")
        
        # Set to eval mode
        model.eval()
        
        return cls(model, processor, device)
    
    def reset_state(self, batch_size: int = 1):
        """
        Reset the memory state for a new episode.
        
        Args:
            batch_size: Batch size for inference
        """
        self.current_memory = self.model.init_memory(batch_size, self.device)
        self.batch_size = batch_size
        print(f"Memory state reset (batch_size={batch_size})")
    
    @torch.no_grad()
    def step(
        self,
        image,
        text: str,
        reset_state: bool = False,
    ) -> Tuple[str, Dict]:
        """
        Process a single step with stateful inference.
        
        Args:
            image: PIL Image or tensor
            text: Text prompt
            reset_state: Whether to reset memory before this step
        
        Returns:
            generated_text: Generated response/action
            info: Dictionary with memory and logits
        """
        if reset_state or self.current_memory is None:
            self.reset_state(batch_size=1)
        
        # Process inputs with processor
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
        )
        
        # Move to device
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                  for k, v in inputs.items()}
        
        # Forward pass with memory
        outputs = self.model(
            input_ids=inputs.get('input_ids'),
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            prev_memory=self.current_memory,
            attention_mask=inputs.get('attention_mask'),
        )
        
        # Update memory for next step
        self.current_memory = outputs.new_memory
        
        # Generate text from logits using base model's generate
        # We need to prepare inputs with the updated memory
        generated_ids = self._generate_with_memory(
            inputs=inputs,
            memory=self.current_memory,
        )
        
        # Decode
        generated_text = self.processor.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        
        info = {
            'memory': outputs.new_memory,
            'raw_memory_hidden': outputs.raw_memory_hidden,
            'logits': outputs.logits,
        }
        
        return generated_text, info
    
    def _generate_with_memory(
        self,
        inputs: Dict[str, torch.Tensor],
        memory: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> torch.Tensor:
        """
        Generate text with memory context.

        Matches the training layout exactly:
            [memory_token] + [visual_patch_embeddings] + [text_embeddings]

        Bugs fixed vs the original:
          1. Memory was APPENDED at training but APPENDED here — now PREPENDED
             to match StatefulJarvisVLA._build_inputs_with_memory.
          2. pixel_values were never fed through the visual encoder — images
             were silently dropped during generation.
          3. Attention mask was extended by appending (wrong) — now prepended.
        """
        # Project memory to hidden dimension
        projected_memory = self.model.memory_projections.project_in(memory)

        # Embed text tokens
        if hasattr(self.model.base_model, 'model') and hasattr(self.model.base_model.model, 'embed_tokens'):
            embed_layer = self.model.base_model.model.embed_tokens
        else:
            embed_layer = self.model.base_model.get_input_embeddings()

        input_ids = inputs['input_ids'].to(embed_layer.weight.device)
        text_embeds = embed_layer(input_ids)

        # Replace image-placeholder positions with real patch embeddings,
        # identical to what _build_inputs_with_memory does during training.
        pixel_values = inputs.get('pixel_values')
        image_grid_thw = inputs.get('image_grid_thw')
        if pixel_values is not None:
            pixel_values = pixel_values.to(
                device=embed_layer.weight.device,
                dtype=self.model.base_model.visual.get_dtype(),
            )
            image_embeds = self.model.base_model.visual(pixel_values, grid_thw=image_grid_thw)
            image_mask = (input_ids == self.model.base_model.config.image_token_id)
            text_embeds = text_embeds.clone()
            text_embeds[image_mask] = image_embeds.to(text_embeds.dtype)

        # PREPEND memory token — training layout: [memory, visual, text]
        memory_embeds = projected_memory.to(
            device=text_embeds.device, dtype=text_embeds.dtype
        ).unsqueeze(1)                               # [batch, 1, hidden_dim]
        inputs_embeds = torch.cat([memory_embeds, text_embeds], dim=1)

        # PREPEND 1 to attention mask for the memory token (matches training)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            memory_mask = torch.ones(
                attention_mask.shape[0], 1,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat([memory_mask, attention_mask], dim=1)

        # Generate
        outputs = self.model.base_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            use_cache=True,
        )

        return outputs
    
    @torch.no_grad()
    def generate(
        self,
        image,
        text: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
        reset_state: bool = False,
    ) -> str:
        """
        Generate a response with stateful context.

        This is the main inference interface.

        Fix vs original:
          The original ran TWO forward passes: one for generation (with stale
          memory) and one to update memory (result ignored for the generation
          that already happened).  Now: one forward pass updates the memory
          from the current observation, then generation uses the updated state.
          This matches the step() method and the training-time semantics.

        Args:
            image: PIL Image
            text: Text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample
            reset_state: Whether to reset memory before generation

        Returns:
            generated_text: The generated response
        """
        if reset_state:
            self.reset_state(batch_size=1)

        inputs = self.processor(text=text, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        if self.current_memory is None:
            self.reset_state(batch_size=1)

        # Single forward pass: observe the current frame, update memory.
        # Generation then uses the freshly-updated state (same as step()).
        fwd = self.model(
            input_ids=inputs.get('input_ids'),
            pixel_values=inputs.get('pixel_values'),
            image_grid_thw=inputs.get('image_grid_thw'),
            prev_memory=self.current_memory,
            attention_mask=inputs.get('attention_mask'),
        )
        self.current_memory = fwd.new_memory

        # Generate text conditioned on the updated memory
        generated_ids = self._generate_with_memory(
            inputs=inputs,
            memory=self.current_memory,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
        )

        generated_text = self.processor.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True,
        )
        return generated_text


def interactive_inference_demo(
    model_path: str,
    video_path: Optional[str] = None,
):
    """
    Run an interactive demo of stateful inference.
    
    Args:
        model_path: Path to model checkpoint
        video_path: Optional path to video file for testing
    """
    print("Loading model...")
    inference = StatefulVLAInference.from_pretrained(model_path)
    
    print("\n" + "="*60)
    print("Stateful VLA Inference Demo")
    print("="*60)
    print("Commands:")
    print("  'reset' - Reset memory state")
    print("  'inventory' - Ask about inventory contents")
    print("  'status' - Show current memory state info")
    print("  'quit' - Exit")
    print("="*60)
    
    import cv2
    
    # Load video if provided
    cap = None
    if video_path and Path(video_path).exists():
        cap = cv2.VideoCapture(video_path)
        print(f"\nLoaded video: {video_path}")
    
    frame_idx = 0
    inference.reset_state()
    
    while True:
        print(f"\n[Frame {frame_idx}]")
        
        # Get user input
        user_input = input("You: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        if user_input.lower() == 'reset':
            inference.reset_state()
            print("Memory state reset.")
            frame_idx = 0
            if cap is not None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        
        if user_input.lower() == 'status':
            mem_norm = torch.norm(inference.current_memory).item()
            print(f"Memory state: shape={inference.current_memory.shape}, norm={mem_norm:.4f}")
            continue
        
        # Get frame
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                print("End of video, looping...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            image = Image.fromarray(frame)
            frame_idx += 1
        else:
            # Create dummy frame
            from PIL import Image
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Generate response
        if user_input.lower() == 'inventory':
            prompt = "What items do you currently have in your inventory? List them."
        else:
            prompt = user_input
        
        print("Model thinking...")
        response = inference.generate(
            image=image,
            text=prompt,
            max_new_tokens=100,
            do_sample=False,
        )
        
        print(f"Model: {response}")
    
    if cap is not None:
        cap.release()
    
    print("\nDemo complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Stateful VLA Inference")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Path to video file for testing",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive demo",
    )
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_inference_demo(args.model_path, args.video_path)
    else:
        # Simple test
        print("Loading model...")
        inference = StatefulVLAInference.from_pretrained(args.model_path)
        print("Model loaded successfully!")
        mem = inference.model.init_memory(1, inference.device)
        print(f"Initial memory: shape={mem.shape}, norm={torch.norm(mem).item():.4f}")
