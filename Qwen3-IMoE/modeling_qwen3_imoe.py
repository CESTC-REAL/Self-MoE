import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from transformers import (
    GradientCheckpointingLayer,
    Cache,
    PreTrainedModel,
    DynamicCache,
    GenerationMixin,
)
from transformers.activations import ACT2FN
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple
from transformers.modeling_outputs import MoeModelOutputWithPast, MoeCausalLMOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeConfig,
    Qwen3MoeAttention,
    Qwen3MoeMLP, Qwen3MoeRMSNorm, Qwen3MoeRotaryEmbedding,
    load_balancing_loss_func,
)
from transformers.processing_utils import Unpack
from transformers.utils.generic import OutputRecorder, check_model_inputs

# CS-MoE / IMoE: Cross-layer Shared Mixture-of-Experts on the Qwen3-MoE backbone.
#
# Architecture (paper Sec. 2):
#   * A single global shared expert pool (Qwen3MoeSharedExperts) is instantiated
#     once and referenced by every layer.
#   * Each layer owns (num_experts - num_shared_experts) / num_hidden_layers
#     local independent experts (Fixed Path): activated for every token with a
#     constant gate of 1.0, no routing.
#   * A layer-specific router (Qwen3MoeTopKRouter) routes every token to
#     top_k = num_experts_per_tok - num_local_experts experts of the shared
#     pool (Dynamic Path).


try:
    from transformers.utils import is_grouped_mm_available
except ImportError:
    def is_grouped_mm_available():
        return False


class Qwen3MoeSharedExperts(nn.Module):
    """共享的专家权重池，供多个层使用"""

    def __init__(self, config, shared_experts=None):
        super().__init__()
        self.num_experts = config.num_shared_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        if shared_experts is not None:
            # 使用共享的专家权重
            self.register_parameter('gate_up_proj', shared_experts.gate_up_proj)
            self.register_parameter('down_proj', shared_experts.down_proj)
        else:
            # 创建新的专家权重
            self.register_parameter(
                'gate_up_proj',
                nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
            )
            self.register_parameter(
                'down_proj',
                nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
            )

        self.act_fn = ACT2FN[config.hidden_act]

    def reset_parameters(self):
        """初始化专家权重"""
        init.xavier_uniform_(self.gate_up_proj)
        init.xavier_uniform_(self.down_proj)


class Qwen3InflateMoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""
    def __init__(self, config, shared_experts=None):
        super().__init__()
        self.num_shared_experts = config.num_shared_experts
        self.num_local_experts = int((config.num_experts - config.num_shared_experts) / config.num_hidden_layers)
        self.num_experts = self.num_shared_experts + self.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size

        # 创建独立专家权重
        self.local_gate_up_proj = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.local_down_proj = nn.Parameter(torch.empty(self.num_local_experts, self.hidden_dim, self.intermediate_dim))

        # 存储共享专家
        self.shared_experts = shared_experts
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        # 边界检查 - top_k_index 只包含共享专家的索引
        assert torch.all((top_k_index >= 0) & (top_k_index < self.num_experts)), \
            f"Invalid shared expert indices detected: min={top_k_index.min()}, max={top_k_index.max()}"
        assert top_k_weights.shape == top_k_index.shape, \
            f"Shape mismatch: top_k_weights {top_k_weights.shape}, top_k_index {top_k_index.shape}"

        final_hidden_states = torch.zeros_like(hidden_states)

        with torch.no_grad():
            # 处理共享专家的激活掩码
            if self.num_shared_experts > 0:
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_shared_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit_shared = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            # 添加所有本地专家到 expert_hit
            if self.num_local_experts > 0:
                local_experts_range = torch.arange(self.num_local_experts, device=hidden_states.device).unsqueeze(-1)
                if self.num_shared_experts > 0:
                    expert_hit = torch.cat([expert_hit_shared, local_experts_range + self.num_shared_experts], dim=0)
                else:
                    expert_hit = local_experts_range

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx >= self.num_experts:
                continue

            # 判断是否为本地专家（ID >= num_shared_experts）
            if expert_idx >= self.num_shared_experts:
                # 本地专家：对所有 token 激活，权重为 1
                token_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
                current_state = hidden_states[token_idx]
                local_expert_idx = expert_idx - self.num_shared_experts
                gate, up = nn.functional.linear(current_state, self.local_gate_up_proj[local_expert_idx]).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = nn.functional.linear(current_hidden_states, self.local_down_proj[local_expert_idx])

                # 数值稳定性处理
                current_hidden_states = torch.clamp(current_hidden_states, min=-1e6, max=1e6)
                # 本地专家权重设为 1
                current_hidden_states = current_hidden_states * 1.0
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
            else:
                # 共享专家：根据 top-k 路由选择
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate, up = nn.functional.linear(current_state, self.shared_experts.gate_up_proj[expert_idx]).chunk(2, dim=-1)
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = nn.functional.linear(current_hidden_states, self.shared_experts.down_proj[expert_idx])

                # 数值稳定性处理
                current_hidden_states = torch.clamp(current_hidden_states, min=-1e6, max=1e6)
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


class Qwen3MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 只对共享专家进行路由，top_k 的值根据共享专家数量和隐藏层数量动态计算
        self.top_k = config.num_experts_per_tok - int((config.num_experts - config.num_shared_experts) / config.num_hidden_layers)
        # self.num_experts = int((config.num_experts - config.num_shared_experts) / config.num_hidden_layers) + config.num_shared_experts
        self.num_experts = config.num_shared_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # Raw scores over the shared pool, as in the stock Qwen3-MoE router.
        # The aux load-balancing loss consumes these raw logits, while gating
        # uses the renormalized Top-K scores below.
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        gate_scores = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(gate_scores, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_scores = router_top_value.to(router_logits.dtype)
        return router_logits, router_scores, router_indices


class Qwen3SparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig, shared_experts=None):
        super().__init__()
        self.experts = Qwen3InflateMoeExperts(config, shared_experts=shared_experts)
        self.gate = Qwen3MoeTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)


class Qwen3MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3MoeConfig, layer_idx: int, shared_experts=None):
        super().__init__()
        self.layer_idx = layer_idx
        self.self_attn = Qwen3MoeAttention(config, layer_idx)

        if (layer_idx not in config.mlp_only_layers) and (
                config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            # 使用共享专家（如果提供）
            self.mlp = Qwen3SparseMoeBlock(config, shared_experts=shared_experts)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            use_cache: bool | None = False,
            cache_position: torch.LongTensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


@auto_docstring
class Qwen3MoePreTrainedModel(PreTrainedModel):
    config: Qwen3MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3MoeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _can_compile_fullgraph = (
        is_grouped_mm_available()
    )  # https://huggingface.co/docs/transformers/experts_interface#torchcompile
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(Qwen3MoeTopKRouter, layer_name="mlp.gate", index=0),
        "hidden_states": Qwen3MoeDecoderLayer,
        "attentions": Qwen3MoeAttention,
    }

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, Qwen3InflateMoeExperts):
            init.normal_(module.local_gate_up_proj, mean=0.0, std=std)
            init.normal_(module.local_down_proj, mean=0.0, std=std)
        elif isinstance(module, Qwen3MoeSharedExperts):
            # Without this, the shared pool stays as uninitialized
            # `torch.empty` memory (latent NaN/garbage weights).
            module.reset_parameters()
        elif isinstance(module, Qwen3MoeTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)


@auto_docstring
class Qwen3MoeModel(Qwen3MoePreTrainedModel):
    def get_tied_weights_keys(self):
        """Layer-side aliases of the global shared pool.

        Every layer references the same parameter tensors as
        ``model.shared_experts`` (sharing is established at construction
        time). Listing the layer-side keys here tells ``save_pretrained`` to
        store each shared tensor only once (under ``model.shared_experts.*``).
        """
        tied_weights_keys = []
        for i in range(self.config.num_hidden_layers):
            # Relative to this module (the framework prepends the "model." prefix)
            tied_weights_keys.append(f"layers.{i}.mlp.experts.shared_experts.gate_up_proj")
            tied_weights_keys.append(f"layers.{i}.mlp.experts.shared_experts.down_proj")
        return tied_weights_keys

    # @check_model_inputs
    # @auto_docstring
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        # 创建共享专家池
        if config.num_shared_experts > 0:
            self.shared_experts = Qwen3MoeSharedExperts(config)
        else:
            self.shared_experts = None

        self.num_hidden_layers = config.num_hidden_layers
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, layer_idx, shared_experts=self.shared_experts) for layer_idx in
             range(config.num_hidden_layers)]
        )
        self.norm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self._tied_weights_keys = self.get_tied_weights_keys()

        # Initialize weights and apply final processing
        self.post_init()

    @check_model_inputs
    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            use_cache: bool | None = None,
            cache_position: torch.LongTensor | None = None,
            **kwargs: Unpack[TransformersKwargs],
    ) -> MoeModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
        causal_mask = mask_function(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)

        return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class Qwen3InflateMoeForCausalLM(Qwen3MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        # Routing (and therefore the aux loss) only spans the shared pool:
        # num_experts for the loss must be M, and top_k must exclude the
        # layer-local experts that are activated unconditionally.
        self.num_shared_experts = config.num_shared_experts
        self.num_local_experts = int(
            (config.num_experts - config.num_shared_experts) / config.num_hidden_layers
        )
        self.num_routed_experts_per_tok = self.num_experts_per_tok - self.num_local_experts

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor | None = None,
            attention_mask: torch.Tensor | None = None,
            position_ids: torch.LongTensor | None = None,
            past_key_values: Cache | None = None,
            inputs_embeds: torch.FloatTensor | None = None,
            labels: torch.LongTensor | None = None,
            use_cache: bool | None = None,
            output_router_logits: bool | None = None,
            cache_position: torch.LongTensor | None = None,
            logits_to_keep: int | torch.Tensor = 0,
            **kwargs: Unpack[TransformersKwargs],
    ) -> MoeCausalLMOutputWithPast:

        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: MoeModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_router_logits=output_router_logits,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, self.vocab_size, **kwargs)

        aux_loss = None
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_shared_experts,
                self.num_routed_experts_per_tok,
                attention_mask,
            )
            if labels is not None:
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

        return MoeCausalLMOutputWithPast(
            loss=loss,
            aux_loss=aux_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )


if __name__ == "__main__":
    """Smoke test: build the model from a config in configs/, run a forward
    pass and print the physical vs. activated parameter counts.

        python modeling_qwen3_imoe.py --config configs/qwen3_0_6_A0_6b.json
    """
    import argparse

    import torch
    from transformers import AutoConfig

    parser = argparse.ArgumentParser(description="Smoke test for the CS-MoE (IMoE) model")
    parser.add_argument("--config", type=str, default="configs/qwen3_0_6_A0_6b.json")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.config, trust_remote_code=True)
    model = Qwen3InflateMoeForCausalLM(config=config)

    total = sum(p.numel() for p in model.parameters())
    n_local = int((config.num_experts - config.num_shared_experts) / config.num_hidden_layers)
    expert_dim = 3 * config.hidden_size * config.moe_intermediate_size
    activated_ffn = config.num_hidden_layers * (n_local + config.num_experts_per_tok) * expert_dim
    print(f"Total parameters:      {total / 1e9:.2f}B")
    print(f"Activated FFN params:  {activated_ffn / 1e9:.2f}B "
          f"(+ attention & embeddings; {config.num_experts_per_tok} experts/layer "
          f"= {n_local} local + {config.num_experts_per_tok - n_local} routed shared)")

    input_ids = torch.randint(0, config.vocab_size, (1, 16))
    labels = input_ids.clone()
    outputs = model(input_ids=input_ids, labels=labels, output_router_logits=True)
    print(f"logits: {tuple(outputs.logits.shape)} | loss: {outputs.loss.item():.4f} "
          f"| aux_loss: {outputs.aux_loss.item():.4f}")
