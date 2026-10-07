# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.deepseek_mtp import (
    DeepSeekMTP,
    DeepSeekMultiTokenPredictor,
    DeepSeekMultiTokenPredictorLayer,
    SharedHead,
)
from vllm.model_executor.models.utils import maybe_prefix

from .model import Dots3NoteDecoderLayer


class Dots3NoteMultiTokenPredictorLayer(DeepSeekMultiTokenPredictorLayer):
    def __init__(self, vllm_config: VllmConfig, prefix: str) -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.enorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hnorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.eh_proj = ReplicatedLinear(
            config.hidden_size * 2,
            config.hidden_size,
            bias=False,
            quant_config=vllm_config.quant_config,
            prefix=f"{prefix}.eh_proj",
        )
        self.shared_head = SharedHead(config=config, prefix=prefix, quant_config=vllm_config.quant_config)
        self.mtp_block = Dots3NoteDecoderLayer(vllm_config=vllm_config, config=config, prefix=prefix)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_index: int = 0,
    ):
        assert inputs_embeds is not None
        inputs_embeds = torch.where(positions.unsqueeze(-1) == 0, 0, inputs_embeds)
        inputs_embeds = self.enorm(inputs_embeds)
        previous_hidden_states = self.hnorm(previous_hidden_states)
        hidden_states, _ = self.eh_proj(torch.cat([inputs_embeds, previous_hidden_states], dim=-1))
        hidden_states, residual = self.mtp_block(positions=positions, hidden_states=hidden_states, residual=None)
        hidden_states = residual + hidden_states
        if self.mtp_block.use_sequence_parallel_moe:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, 0)
            hidden_states = hidden_states[: positions.shape[0]]
        # vLLM 0.30 recycles post-norm states and computes logits from pre-norm states.
        return hidden_states, self.shared_head(hidden_states)


class Dots3NoteMultiTokenPredictor(DeepSeekMultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        assert vllm_config.speculative_config is not None
        config = vllm_config.speculative_config.draft_model_config.hf_config
        self.mtp_start_layer_idx = config.num_hidden_layers
        # Dots3 uses one full-sharing NextN block recursively for every draft
        # position.  ``num_nextn_predict_layers`` is the number of proposed
        # tokens in generic DeepSeek configs, not the number of independent
        # Dots3 weight sets.  SGLang's reference implementation likewise
        # constructs exactly one head; creating three layers leaves later
        # steps with unmatched/uninitialized weights and destroys acceptance.
        self.num_mtp_layers = 1
        self.layers = nn.ModuleDict(
            {
                str(idx): Dots3NoteMultiTokenPredictorLayer(vllm_config, f"{prefix}.layers.{idx}")
                for idx in range(self.mtp_start_layer_idx, self.mtp_start_layer_idx + self.num_mtp_layers)
            }
        )
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size, prefix=maybe_prefix(prefix, "embed_tokens")
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)


@support_torch_compile
class Dots3NoteMTP(DeepSeekMTP):
    has_own_embed_tokens = True
    has_own_lm_head = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        self.config = vllm_config.model_config.hf_config
        self.quant_config = vllm_config.quant_config
        self.model = Dots3NoteMultiTokenPredictor(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.set_moe_parameters()

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def mtp_weights():
            for name, weight in weights:
                # DeepSeek loads top-level embeddings too. Dots3 has a
                # dedicated draft embedding, which must survive any shard order.
                if name.startswith("model.embed_tokens."):
                    continue
                if name.startswith("model.mtp.embed_tokens."):
                    name = name.replace(
                        "model.mtp.embed_tokens.",
                        f"model.layers.{self.config.num_hidden_layers}.embed_tokens.",
                        1,
                    )
                yield name, weight

        return super().load_weights(mtp_weights())
