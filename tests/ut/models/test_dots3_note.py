# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn
from vllm import ModelRegistry
from vllm.model_executor.models import deepseek_v2
from vllm.transformers_utils.configs.dots3_note import Dots3NoteConfig

from vllm_ascend.models import register_model
from vllm_ascend.models.dots3_note.audio import prepare_audio_features
from vllm_ascend.models.dots3_note.model import Dots3NoteModel, get_dots3_note_layer_config
from vllm_ascend.models.dots3_note.multimodal import (
    Dots3NoteForCausalLM,
    Dots3NoteProcessingInfo,
    DotsMoEVisionTransformer,
    DotsVisionAttention,
    MoESwiGLUFFN,
)


@pytest.mark.parametrize("tp_size", [1, 8])
def test_vision_attention_preserves_local_packed_qkv(tp_size):
    heads, width, tokens = 24, 4, 2
    full = torch.arange(tokens * 3 * heads * width).reshape(tokens, 1, 3, heads, width).float()
    for rank in range(tp_size):
        q, k, v = full[:, :, :, rank * (heads // tp_size) : (rank + 1) * (heads // tp_size)].unbind(2)
        attention = MagicMock(return_value=v.permute(1, 0, 2, 3))
        module = SimpleNamespace(
            qkv=MagicMock(return_value=(torch.cat([q.flatten(-2), k.flatten(-2), v.flatten(-2)], -1), None)),
            q_norm=None,
            k_norm=None,
            num_attention_heads_per_partition=heads // tp_size,
            hidden_size_per_attention_head=width,
            apply_rotary_emb=lambda x, cos, sin: x,
            attn=attention,
            proj=lambda x: (x, None),
        )
        DotsVisionAttention.forward(
            module, torch.zeros(tokens, heads * width), torch.tensor([0, tokens]), torch.zeros(tokens, width)
        )
        actual = attention.call_args.kwargs
        for name, expected in zip(("query", "key", "value"), (q, k, v)):
            torch.testing.assert_close(actual[name], expected.permute(1, 0, 2, 3))


@pytest.mark.parametrize("use_data_parallel", [False, True])
def test_vision_moe_tp_uses_shared_routes_and_weights(use_data_parallel):
    module = nn.Module()
    module.num_routed, module.capacity_factor = 3, 2
    module.router_scoring_func, module.router_scale = "sigmoid", 1.0
    module.use_data_parallel = use_data_parallel
    module.router_bias = torch.zeros(3)
    module.gate_weight = torch.tensor([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    module.experts = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
    for expert, value in zip(module.experts, [1, 3, 8]):
        nn.init.zeros_(expert.weight)
        nn.init.constant_(expert.bias, value)

    def broadcast(tensor, src):
        assert src == 0
        tensor.copy_(torch.tensor([[0, 1], [0, 1]]) if tensor.dtype == torch.int64 else torch.tensor([[0.2, 0.8]] * 2))
        return tensor

    group = MagicMock()
    group.broadcast.side_effect = broadcast
    with (
        patch("vllm_ascend.models.dots3_note.multimodal.get_tensor_model_parallel_world_size", return_value=2),
        patch("vllm_ascend.models.dots3_note.multimodal.get_tp_group", return_value=group),
    ):
        actual = MoESwiGLUFFN.forward(module, torch.ones(2, 2))
    weights = torch.tensor([0.0, 2.0]).sigmoid()
    expected = float((weights * torch.tensor([3.0, 8.0])).sum() / weights.sum()) if use_data_parallel else 2.6
    torch.testing.assert_close(actual, torch.full((2, 2), expected))
    assert group.broadcast.call_count == (0 if use_data_parallel else 2)


def test_dots3_note_uses_upstream_config_and_processor():
    config = Dots3NoteConfig()

    assert config.model_type == "dots3_note"
    assert Dots3NoteProcessingInfo.__bases__[0].__module__.endswith("_common.processor")


def test_tp_multimodal_embeddings_share_rank_zero_values_in_input_order():
    image, audio = torch.ones(2, 4), torch.zeros(3, 4)
    model = SimpleNamespace(
        use_data_parallel=False,
        _process_image_input=lambda *args: (image,),
        _process_audio_input=lambda *args: (audio,),
    )
    with (
        patch("vllm_ascend.models.dots3_note.multimodal.get_tensor_model_parallel_world_size", return_value=2),
        patch("vllm_ascend.models.dots3_note.multimodal.get_tp_group") as group,
    ):
        result = Dots3NoteForCausalLM.embed_multimodal(
            model,
            audio_values=torch.empty(1),
            audio_lengths=torch.tensor([1]),
            pixel_values=torch.empty(1),
            image_grid_thw=torch.tensor([[1, 2, 2]]),
        )
    assert result[0] is audio and result[1] is image
    calls = group.return_value.broadcast.call_args_list
    assert len(calls) == 2 and calls[0].args[0] is audio and calls[1].args[0] is image
    assert all(call.kwargs == {"src": 0} for call in calls)


def test_dots3_note_models_are_registered_to_ascend_implementations():
    register_model()

    assert ModelRegistry.models["Dots3NoteForCausalLM"].module_name == ("vllm_ascend.models.dots3_note")
    assert ModelRegistry.models["Dots3NoteMTPModel"].module_name == ("vllm_ascend.models.dots3_note.mtp")


def test_dots3_note_audio_features_follow_checkpoint_contract():
    config = SimpleNamespace(
        chunk_seconds=60,
        merge_factor=1,
        sampling_rate=16000,
    )

    outputs = prepare_audio_features([torch.zeros(16000)], config)

    assert outputs["audio_features"].shape == (1, 128, 6000)
    assert outputs["audio_sample_lens"].tolist() == [16000]
    assert outputs["audio_segment_counts"].tolist() == [1]
    assert outputs["audio_token_lengths"].tolist() == [13]


def test_dots3_note_vision_weight_loader_requires_all_packed_shards():
    module = nn.Module()
    module.fc13 = nn.Linear(2, 4, bias=False)
    loaded_shards = []
    module.fc13.weight.weight_loader = lambda param, weight, shard_id: loaded_shards.append(shard_id)

    loaded = DotsMoEVisionTransformer.load_weights(
        module,
        [
            ("fc1.weight", torch.ones(2, 2)),
            ("fc3.weight", torch.ones(2, 2)),
        ],
    )

    assert loaded == {"fc13.weight"}
    assert loaded_shards == [0, 1]


def test_dots3_note_weight_mapper_routes_all_submodels():
    weights = [
        ("model.layers.0.weight", torch.tensor(0)),
        ("lm_head.weight", torch.tensor(1)),
        ("vision_encoder.patch_embed.weight", torch.tensor(2)),
        ("audio_encoder.conv.weight", torch.tensor(3)),
    ]

    mapped = list(Dots3NoteForCausalLM.hf_to_vllm_mapper.apply(weights))

    assert [name for name, _ in mapped] == [
        "language_model.model.layers.0.weight",
        "language_model.lm_head.weight",
        "visual.patch_embed.weight",
        "audio_tower.conv.weight",
    ]


def test_dots3_note_projects_sliding_layer_config():
    config = SimpleNamespace(
        num_hidden_layers=2,
        num_attention_heads=16,
        q_lora_rank=1024,
        index_topk=2048,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kv_lora_rank=512,
        attention_gate_type="headwise",
        swa_num_attention_heads=8,
        swa_q_lora_rank=768,
        swa_qk_nope_head_dim=64,
        swa_qk_rope_head_dim=32,
        swa_v_head_dim=96,
        swa_attention_gate_type="elementwise",
        swa_kv_lora_rank=256,
        sliding_window_size=512,
        rope_theta=10_000,
        swa_rope_theta=1_000,
        layer_types=["full_attention", "sliding_attention"],
        moe_layer_freq=1,
        n_routed_experts=8,
    )

    projected = get_dots3_note_layer_config(config, 1)

    assert projected is not config
    assert projected.num_attention_heads == 8
    assert not hasattr(projected, "index_topk")
    assert projected.q_lora_rank == 768
    assert projected.qk_nope_head_dim == 64
    assert projected.qk_rope_head_dim == 32
    assert projected.v_head_dim == 96
    assert projected.kv_lora_rank == 256
    assert projected.attention_gate_type == "elementwise"
    assert projected.rope_parameters == {"rope_type": "default", "rope_theta": 1_000}
    assert projected.n_routed_experts == 8


def test_dots3_note_main_model_filters_mtp_weights(monkeypatch):
    model = Dots3NoteModel.__new__(Dots3NoteModel)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1)
    captured: list[str] = []

    def load_weights(_self, weights):
        captured.extend(name for name, _ in weights)
        return set(captured)

    monkeypatch.setattr(deepseek_v2.DeepseekV2Model, "load_weights", load_weights)
    result = model.load_weights(
        [
            ("model.layers.0.input_layernorm.weight", torch.ones(1)),
            ("model.layers.2.self_attn.q_proj.weight", torch.ones(1)),
            ("model.mtp.embed_tokens.weight", torch.ones(1)),
            ("model.norm.weight", torch.ones(1)),
        ]
    )

    assert result == {
        "model.layers.0.input_layernorm.weight",
        "model.norm.weight",
    }


def test_dots3_note_mtp_uses_dense_sliding_decoder(monkeypatch):
    from vllm_ascend.models.dots3_note import model as dots_model

    config = Dots3NoteConfig(
        num_hidden_layers=2,
        hidden_size=8,
        intermediate_size=16,
        layer_types=["full_attention", "sliding_attention"],
        sliding_window_size=512,
        swa_num_attention_heads=4,
        swa_q_lora_rank=8,
        swa_kv_lora_rank=16,
        swa_qk_nope_head_dim=8,
        swa_qk_rope_head_dim=4,
        swa_v_head_dim=4,
        swa_attention_gate_type="headwise",
        swa_rope_theta=50000,
        index_topk=8,
        moe_layer_freq=1,
    )
    config.layer_types = [*config.layer_types, "sliding_attention"]
    selected = {}

    def attention(**kwargs):
        selected["attention"] = kwargs
        return nn.Identity()

    def dense(**kwargs):
        selected["dense"] = kwargs
        return nn.Identity()

    monkeypatch.setattr(dots_model, "RMSNorm", lambda *args, **kwargs: nn.Identity())
    monkeypatch.setattr(dots_model, "Dots3NoteAttention", attention)
    monkeypatch.setattr(dots_model, "DeepseekV2MLP", dense)
    monkeypatch.setattr(
        dots_model, "DeepseekV2MoE", lambda **kwargs: (_ for _ in ()).throw(AssertionError("MTP must be dense"))
    )
    dots_model.Dots3NoteDecoderLayer(
        vllm_config=SimpleNamespace(
            quant_config=None, parallel_config=SimpleNamespace(use_sequence_parallel_moe=False)
        ),
        config=config,
        prefix=f"model.layers.{config.num_hidden_layers}",
    )
    assert selected["attention"]["sliding_window"] == config.sliding_window_size - 1
    assert selected["attention"]["config"].kv_lora_rank == config.swa_kv_lora_rank
    assert selected["dense"]["intermediate_size"] == config.intermediate_size


def test_dots3_note_mtp_maps_dedicated_embedding(monkeypatch):
    from vllm.model_executor.models.deepseek_mtp import DeepSeekMTP

    from vllm_ascend.models.dots3_note.mtp import Dots3NoteMTP

    model = Dots3NoteMTP.__new__(Dots3NoteMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=2)
    captured = []

    def load_weights(self, weights):
        captured.extend(weights)
        return {name for name, _ in captured}

    monkeypatch.setattr(DeepSeekMTP, "load_weights", load_weights)
    dedicated = torch.ones(2, 2)
    loaded = model.load_weights([("model.mtp.embed_tokens.weight", dedicated)])
    assert loaded == {"model.layers.2.embed_tokens.weight"}
    assert captured[0][1] is dedicated
    assert model.has_own_embed_tokens and not model.has_own_lm_head


@pytest.mark.parametrize("dedicated_first", [False, True])
def test_mtp_embedding_survives_target_embedding_shard_order(monkeypatch, dedicated_first):
    from vllm.model_executor.models import deepseek_mtp

    from vllm_ascend.models.dots3_note.mtp import Dots3NoteMTP

    model = Dots3NoteMTP.__new__(Dots3NoteMTP)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=2, num_nextn_predict_layers=1, n_routed_experts=0)
    model.is_fused_shared_expert_enabled = False
    model.model = nn.Module()
    model.model.mtp_start_layer_idx = 2
    model.model.num_mtp_layers = 1
    model.model.embed_tokens = nn.Embedding(2, 2)
    monkeypatch.setattr(deepseek_mtp, "fused_moe_make_expert_params_mapping", lambda *a, **k: [])
    monkeypatch.setattr(deepseek_mtp, "get_pp_missing_layer_names", lambda model: [])
    monkeypatch.setattr(deepseek_mtp, "is_mtp_completeness_check_enabled", lambda: False)
    dedicated = torch.full((2, 2), 7.0)
    weights = [("model.embed_tokens.weight", torch.ones(2, 2)), ("model.mtp.embed_tokens.weight", dedicated)]
    if dedicated_first:
        weights.reverse()
    loaded = model.load_weights(iter(weights))
    assert "model.embed_tokens.weight" in loaded
    torch.testing.assert_close(model.model.embed_tokens.weight, dedicated)


@pytest.mark.parametrize("projection", ["wk", "weights_proj"])
@pytest.mark.parametrize("mx", [False, True])
def test_indexer_load_decodes_both_fp8_projections_in_either_order(projection, mx):
    model = Dots3NoteModel.__new__(Dots3NoteModel)
    prefix = f"model.layers.0.self_attn.indexer.{projection}"
    weight = torch.ones(64, 64).to(torch.float8_e4m3fn)
    scale = torch.full((64, 2), 128, dtype=torch.uint8) if mx else torch.full((1, 1), 2.0)
    suffix = "weight_scale" if mx else "weight_scale_inv"
    pairs = [(f"{prefix}.{suffix}", scale), (f"{prefix}.weight", weight)]
    for ordered in (pairs, pairs[::-1]):
        outputs = list(model._adapt_indexer_weights(ordered))
        assert len(outputs) == 1 and outputs[0][0] == f"{prefix}.weight"
        torch.testing.assert_close(outputs[0][1], torch.full((64, 64), 2.0, dtype=torch.bfloat16))
    with pytest.raises(ValueError, match="Incomplete"):
        list(model._adapt_indexer_weights(pairs[:1]))


def test_mtp_quantized_eh_projection_preserves_v030_hidden_state_contract():
    from vllm_ascend.models.dots3_note.mtp import Dots3NoteMultiTokenPredictorLayer

    class TupleProjection(nn.Module):
        def forward(self, value):
            return value[:, :4] + 2 * value[:, 4:], None

    class Decoder(nn.Module):
        use_sequence_parallel_moe = False

        def forward(self, positions, hidden_states, residual):
            return hidden_states, torch.ones_like(hidden_states)

    layer = Dots3NoteMultiTokenPredictorLayer.__new__(Dots3NoteMultiTokenPredictorLayer)
    nn.Module.__init__(layer)
    layer.enorm = layer.hnorm = nn.Identity()
    layer.eh_proj = TupleProjection()
    layer.mtp_block = Decoder()
    layer.shared_head = lambda value: value / 2
    embeddings = torch.ones(3, 4)
    previous = torch.full((3, 4), 3.0)
    positions = torch.arange(3)
    pre_norm, recycled = layer(torch.arange(3), positions, previous, embeddings)
    expected = torch.full((3, 4), 8.0)
    expected[0] = 7.0
    torch.testing.assert_close(pre_norm, expected)
    torch.testing.assert_close(recycled, expected / 2)
