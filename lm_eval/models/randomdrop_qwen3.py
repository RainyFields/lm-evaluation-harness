import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoConfig, AutoModelForCausalLM
from lm_eval.models.huggingface import HFLM
from lm_eval.api.registry import register_model


class RandomDropWrapper(nn.Module):
    """
    A wrapper for individual transformer layers that optionally applies token dropping
    during inference and saves per-layer hidden states for analysis/debugging.

    Args:
        layer (nn.Module): The transformer layer to wrap.
        drop_ratio (float): Probability of dropping each token (0.0 means keep all).
        layer_id (int): Identifier for the layer (used in saving filenames).
        rotary_emb (callable): Rotary positional embedding function.
        mask_only (bool): Whether to apply masking (True) or pruning (False).
        save_hidden (bool): Whether to save per-layer hidden states to disk.
        hidden_save_dir (str): Directory where hidden state .pt files are saved.
    """
    def __init__(self, layer, drop_ratio=0, layer_id=None, rotary_emb=None, mask_only=False,
                 save_hidden=False, hidden_save_dir=None):
        super().__init__()
        self.layer = layer
        self.drop_ratio = drop_ratio
        self.layer_id = layer_id
        self.rotary_emb = rotary_emb
        self.mask_only = mask_only
        self.save_hidden = save_hidden
        self.hidden_save_dir = hidden_save_dir
        self.input_counter = 0

        if self.save_hidden and self.hidden_save_dir:
            os.makedirs(self.hidden_save_dir, exist_ok=True)

    def forward(self, hidden_states, *args, **kwargs):
        position_embeddings = kwargs.pop("position_embeddings", None)

        if not self.training and self.drop_ratio > 0:
            keep_prob = 1.0 - self.drop_ratio
            B, T, D = hidden_states.shape
            min_tokens = max(1, int(T * 0.1))  # retain at least 10% tokens

            raw_mask = torch.rand(B, T, device=hidden_states.device) < keep_prob
            mask = raw_mask.clone()

            for i in range(B):
                if mask[i].sum() < min_tokens:
                    keep_indices = torch.randperm(T, device=hidden_states.device)[:min_tokens]
                    mask[i].zero_()
                    mask[i][keep_indices] = True

            mask = mask.unsqueeze(-1).to(hidden_states.dtype)
            mask_bool = mask.squeeze(-1).bool()

            if self.mask_only:
                hidden_states = hidden_states * mask
            else:
                new_hidden_states, new_position_ids = [], []
                for i in range(B):
                    pruned = hidden_states[i][mask_bool[i]]
                    new_hidden_states.append(pruned)
                    pos_ids = torch.arange(pruned.shape[0], device=hidden_states.device)
                    new_position_ids.append(pos_ids)

                hidden_states = pad_sequence(new_hidden_states, batch_first=True)
                padded_pos_ids = pad_sequence(new_position_ids, batch_first=True)
                cos, sin = self.rotary_emb(hidden_states, padded_pos_ids)
                position_embeddings = (cos, sin)

        # Save hidden state tensor for analysis (only if prefill stage)
        if self.save_hidden and self.hidden_save_dir and hidden_states.shape[1] > 1:
            filename = (
                f"layer_{self.layer_id:02d}_input_{self.input_counter:05d}"
                f"_dropratio_{self.drop_ratio}_maskonly_{self.mask_only}.pt"
            )
            full_path = os.path.join(self.hidden_save_dir, filename)
            torch.save(hidden_states.detach().cpu(), full_path)
            self.input_counter += 1


        outputs = self.layer(hidden_states, *args, position_embeddings=position_embeddings, **kwargs)

        if isinstance(outputs, tuple):
            hidden_states = outputs[0]
            others = outputs[1:]
        else:
            hidden_states = outputs
            others = ()

        return (hidden_states,) + others if others else (hidden_states,)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.layer, name)


@register_model("qwen3-token-drop")
class Qwen3TokenDropLM(HFLM):
    """
    Custom wrapper for Qwen3-8B model to support layer-level token dropping and 
    per-layer hidden state saving during evaluation.

    Supported model_args:
        pretrained (str): Path or model ID for pretrained Qwen3 model.
        drop_ratio (float): Drop probability per token (default: 0.01).
        drop_layers (str | int | list[int]): Layer(s) to apply drop wrapper. "all" applies to all.
        mask_only (bool): Whether to mask instead of prune tokens (default: False).
        save_hidden (bool): If True, saves per-layer hidden states to disk.
        hidden_save_dir (str): Directory path to store .pt files of hidden states.
        revision, trust_remote_code, subfolder, parallelize, etc: standard HuggingFace args.

    Example CLI usage:
        lm_eval \
          --model qwen3-token-drop \
          --model_args pretrained=/path/to/model,drop_ratio=0.01,drop_layers=all,mask_only=False,save_hidden=True,hidden_save_dir=/output/path \
          --tasks niah_single_2 \
          --batch_size 1
    """
    def _create_model(self, **kwargs):
        pretrained = kwargs.pop("pretrained")
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        revision = kwargs.pop("revision", "main")
        subfolder = kwargs.pop("subfolder", "")
        drop_ratio = float(kwargs.pop("drop_ratio", 0.01))
        mask_only = kwargs.pop("mask_only", False)
        drop_layers = kwargs.pop("drop_layers", "all")

        save_hidden = kwargs.pop("save_hidden", False)
        hidden_save_dir = kwargs.pop("hidden_save_dir", None)

        if isinstance(drop_layers, str) and drop_layers != "all":
            drop_layers = [int(drop_layers)]
        elif isinstance(drop_layers, int):
            drop_layers = [drop_layers]
        elif drop_layers == "all" or drop_layers is None:
            drop_layers = "all"
        elif isinstance(drop_layers, list):
            drop_layers = [int(i) for i in drop_layers]

        config = AutoConfig.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
            subfolder=subfolder,
        )

        accelerate_args = self._get_accelerate_args(
            parallelize=kwargs.pop("parallelize", False),
            device_map=kwargs.pop("device_map", None),
            max_memory_per_gpu=kwargs.pop("max_memory_per_gpu", None),
            max_cpu_memory=kwargs.pop("max_cpu_memory", None),
            offload_folder=kwargs.pop("offload_folder", "./offload"),
            gpus=torch.cuda.device_count(),
        )

        self._model = AutoModelForCausalLM.from_pretrained(
            pretrained,
            config=config,
            revision=revision,
            trust_remote_code=trust_remote_code,
            subfolder=subfolder,
            torch_dtype=torch.float16,
            **accelerate_args,
        )

        if hasattr(self._model, "model") and hasattr(self._model.model, "layers"):
            rotary_emb = self._model.model.rotary_emb
            for i, layer in enumerate(self._model.model.layers):
                if drop_layers == "all" or i in drop_layers:
                    self._model.model.layers[i] = RandomDropWrapper(
                        layer=layer,
                        drop_ratio=drop_ratio,
                        layer_id=i,
                        rotary_emb=rotary_emb,
                        mask_only=mask_only,
                        save_hidden=save_hidden,
                        hidden_save_dir=hidden_save_dir,
                    )
        else:
            raise ValueError("Expected model.model.layers in Qwen architecture.")
