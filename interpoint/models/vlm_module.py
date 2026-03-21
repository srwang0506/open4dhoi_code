"""
Vision-Language Model (VLM) Module for InterActVLM-Discrete
Uses LLaVA-v1.5 with LoRA for semantic reasoning about HOI contacts
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List

try:
    from transformers import (
        LlavaForConditionalGeneration,
        AutoProcessor,
        BitsAndBytesConfig
    )
    from peft import LoraConfig, get_peft_model, TaskType
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers/peft not available. VLM module will use dummy outputs.")


class VLMModule(nn.Module):
    """
    VLM Module for extracting semantic interaction context.
    
    Uses LLaVA-v1.5 with LoRA fine-tuning to extract embeddings
    from specialized tokens <HCON> (human contact) and <OCON> (object contact).
    
    Architecture:
    - Input: RGB Image + Text Prompt
    - Process: LLaVA encoder + LoRA adapted decoder
    - Output: E_human, E_object semantic embeddings
    """
    
    def __init__(
        self,
        model_name: str = "llava-hf/llava-1.5-7b-hf",
        d_tr: int = 256,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        freeze_backbone: bool = True,
        use_4bit: bool = False,
        device: str = 'cuda',
        device_map: str = 'auto'  # Use 'cuda:0' for single GPU
    ):
        """
        Initialize the VLM module.

        Args:
            model_name: HuggingFace model name for LLaVA
            d_tr: Output embedding dimension
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            freeze_backbone: Whether to freeze the backbone
            use_4bit: Whether to use 4-bit quantization
            device: Compute device
            device_map: Device map for model loading ('auto' or 'cuda:0')
        """
        super().__init__()

        self.model_name = model_name
        self.d_tr = d_tr
        self.device = device
        self.device_map = device_map
        self.use_transformers = TRANSFORMERS_AVAILABLE

        # Special tokens for contact extraction
        self.human_token = "<HCON>"
        self.object_token = "<OCON>"

        if self.use_transformers:
            self._setup_model(
                model_name, lora_r, lora_alpha, lora_dropout,
                freeze_backbone, use_4bit
            )
        else:
            # Dummy projection for testing
            self.hidden_size = 4096
            self.human_proj = nn.Linear(self.hidden_size, d_tr)
            self.object_proj = nn.Linear(self.hidden_size, d_tr)
    
    def _setup_model(
        self,
        model_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
        freeze_backbone: bool,
        use_4bit: bool
    ):
        """Setup LLaVA model with LoRA."""
        # Quantization config
        if use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
        else:
            bnb_config = None
        
        # Load model
        # Use self.device_map to support single-GPU loading ('cuda:0') or auto distribution
        try:
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=self.device_map,
                torch_dtype=torch.bfloat16
            )
        except AttributeError as e:
            if use_4bit:
                print("Warning: 4-bit quantization failed, retrying without quantization.")
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_name,
                    device_map=self.device_map,
                    torch_dtype=torch.bfloat16
                )
            else:
                raise e
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Add special tokens
        special_tokens = {
            'additional_special_tokens': [self.human_token, self.object_token]
        }
        self.processor.tokenizer.add_special_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.processor.tokenizer))
        
        # Get token IDs
        self.human_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.human_token
        )
        self.object_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.object_token
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        if freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
        
        # Apply LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # Get hidden size
        self.hidden_size = self.model.config.text_config.hidden_size
        
        # Projection layers to d_tr
        self.human_proj = nn.Linear(self.hidden_size, self.d_tr)
        self.object_proj = nn.Linear(self.hidden_size, self.d_tr)
    
    def create_prompt(self, task_description: Optional[str] = None) -> str:
        """
        Create the prompt template for HOI reasoning.
        
        Args:
            task_description: Optional task-specific description
            
        Returns:
            Formatted prompt string
        """
        if task_description is None:
            task_description = "Analyze the human-object interaction in this image."
        
        prompt = f"""<image>
{task_description}
Identify the contact regions between the human body and the object.
{self.human_token} represents the human body contact areas.
{self.object_token} represents the object contact areas.
Human contacts: {self.human_token}
Object contacts: {self.object_token}"""
        
        return prompt
    
    def forward(
        self,
        images: torch.Tensor,
        prompts: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass to extract semantic embeddings.
        
        Args:
            images: (B, 3, H, W) RGB images
            prompts: Optional list of prompts (one per image)
            
        Returns:
            E_human: (B, d_tr) human semantic context
            E_object: (B, d_tr) object semantic context
        """
        B = images.shape[0]
        device = images.device
        
        if not self.use_transformers:
            # Return dummy embeddings for testing
            E_human = torch.randn(B, self.d_tr, device=device)
            E_object = torch.randn(B, self.d_tr, device=device)
            return E_human, E_object
        
        # Create prompts if not provided
        if prompts is None:
            prompts = [self.create_prompt()] * B
        
        # Process inputs
        inputs = self.processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        # Forward pass with hidden states
        outputs = self.model(
            **inputs,
            output_hidden_states=True
        )
        
        # Get last layer hidden states
        hidden_states = outputs.hidden_states[-1]  # (B, seq_len, hidden_size)
        
        # Find positions of special tokens
        input_ids = inputs['input_ids']
        
        E_human_list = []
        E_object_list = []
        
        for b in range(B):
            # Find human token position
            human_pos = (input_ids[b] == self.human_token_id).nonzero(as_tuple=True)[0]
            if len(human_pos) > 0:
                human_emb = hidden_states[b, human_pos[-1], :]
            else:
                human_emb = hidden_states[b, -1, :]  # Fallback to last token
            
            # Find object token position
            object_pos = (input_ids[b] == self.object_token_id).nonzero(as_tuple=True)[0]
            if len(object_pos) > 0:
                object_emb = hidden_states[b, object_pos[-1], :]
            else:
                object_emb = hidden_states[b, -1, :]
            
            E_human_list.append(human_emb)
            E_object_list.append(object_emb)
        
        E_human = torch.stack(E_human_list)  # (B, hidden_size)
        E_object = torch.stack(E_object_list)  # (B, hidden_size)
        
        # Project to d_tr
        proj_dtype = self.human_proj.weight.dtype
        E_human = self.human_proj(E_human.to(proj_dtype))  # (B, d_tr)
        E_object = self.object_proj(E_object.to(proj_dtype))  # (B, d_tr)
        
        return E_human, E_object
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Get list of trainable parameters (LoRA + projections)."""
        params = []
        
        if self.use_transformers:
            # LoRA parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    params.append(param)
        
        # Projection parameters
        params.extend(self.human_proj.parameters())
        params.extend(self.object_proj.parameters())
        
        return params
    
    def print_trainable_params(self):
        """Print number of trainable parameters."""
        if self.use_transformers:
            self.model.print_trainable_parameters()
        
        proj_params = sum(
            p.numel() for p in self.human_proj.parameters()
        ) + sum(
            p.numel() for p in self.object_proj.parameters()
        )
        print(f"Projection layers: {proj_params:,} parameters")


class LightweightVLM(nn.Module):
    """
    Lightweight VLM alternative using CLIP for faster inference.
    
    Uses CLIP image encoder + text encoder for cross-modal reasoning.
    """
    
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        d_tr: int = 256,
        device: str = 'cuda'
    ):
        """
        Initialize lightweight VLM.
        
        Args:
            model_name: CLIP model name
            d_tr: Output dimension
            device: Compute device
        """
        super().__init__()
        
        self.d_tr = d_tr
        self.device = device
        
        try:
            from transformers import CLIPModel, CLIPProcessor
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.hidden_size = self.model.config.projection_dim
            self.use_clip = True
        except:
            self.hidden_size = 512
            self.use_clip = False
        
        # Cross-attention for human/object reasoning
        self.human_query = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        self.object_query = nn.Parameter(torch.randn(1, 1, self.hidden_size) * 0.02)
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Projections
        self.human_proj = nn.Linear(self.hidden_size, d_tr)
        self.object_proj = nn.Linear(self.hidden_size, d_tr)
    
    def forward(
        self,
        images: torch.Tensor,
        text_prompts: Optional[List[str]] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            images: (B, 3, H, W) RGB images (may be normalized with ImageNet stats)
            text_prompts: Optional text prompts

        Returns:
            E_human, E_object: (B, d_tr) embeddings
        """
        B = images.shape[0]
        device = images.device

        if not self.use_clip:
            return (
                torch.randn(B, self.d_tr, device=device),
                torch.randn(B, self.d_tr, device=device)
            )

        # Default prompts
        if text_prompts is None:
            text_prompts = ["human contact with object"] * B

        # Unnormalize images if they appear to be normalized (values outside [0, 1])
        # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        if images.min() < 0 or images.max() > 1:
            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
            images = images * std + mean
            images = images.clamp(0, 1)

        # Convert to list of PIL-like format for processor
        # Processor expects list of images or numpy arrays
        images_list = [img.cpu() for img in images]

        # Process with CLIP
        inputs = self.processor(
            text=text_prompts,
            images=images_list,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        outputs = self.model(**inputs)
        
        # Get image features
        image_features = outputs.image_embeds.unsqueeze(1)  # (B, 1, hidden)
        
        # Expand queries
        human_q = self.human_query.expand(B, -1, -1)
        object_q = self.object_query.expand(B, -1, -1)
        
        # Cross attention
        human_out, _ = self.cross_attn(human_q, image_features, image_features)
        object_out, _ = self.cross_attn(object_q, image_features, image_features)
        
        # Project
        E_human = self.human_proj(human_out.squeeze(1))
        E_object = self.object_proj(object_out.squeeze(1))
        
        return E_human, E_object
