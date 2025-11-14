# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only FireRedASR LLM model compatible with HuggingFace weights."""
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, Literal, Optional, Union, List

import torch
import torch.nn as nn
import numpy as np
import re
from transformers import BatchFeature
from typing import List
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.config.speech_to_text import SpeechToTextConfig
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.config.multimodal import BaseDummyOptions
from vllm.multimodal.inputs import (AudioItem, ModalityData,
                                    MultiModalDataDict, MultiModalFieldConfig,
                                    MultiModalKwargsItems, MultiModalInputs,
                                    MultiModalEncDecInputs)
from vllm.multimodal.parse import (AudioProcessorItems, DictEmbeddingItems,
                                   EmbeddingItems, ModalityDataItems, MultiModalDataItems,
                                   MultiModalDataParser)
from vllm.multimodal.processing import (BaseMultiModalProcessor,
                                        EncDecMultiModalProcessor,
                                        BaseProcessingInfo, PromptReplacement,
                                        PromptUpdate, PromptUpdateDetails,
                                        PromptIndexTargets)
from vllm.transformers_utils.tokenizer import encode_tokens
from vllm.multimodal.profiling import BaseDummyInputsBuilder
from vllm.sequence import IntermediateTensors
from vllm.utils.tensor_schema import TensorSchema, TensorShape

from .interfaces import SupportsTranscription, SupportsMultiModal, SupportsPP, MultiModalEmbeddings
from .utils import merge_multimodal_embeddings
from .interfaces_base import VllmModelForTextGeneration
from .utils import (AutoWeightsLoader, init_vllm_registered_model,
                    maybe_prefix)
import os

try:
    from transformers import Qwen2Config
except ImportError:
    # Fallback in case Qwen2Config is not available
    from transformers import AutoConfig
    Qwen2Config = AutoConfig

DEFAULT_SPEECH_TOKEN = "<speech>"


def _load_firered_llm_weights(model_path: str, encoder_path: str = None):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load main model checkpoint
    package = torch.load(model_path, map_location=lambda storage, loc: storage,weights_only=False)
    model_state_dict = package.get("model_state_dict", {})
    args = package.get("args", None)

    # Load encoder checkpoint if provided
    encoder_state_dict = {}
    if encoder_path and os.path.exists(encoder_path):
        encoder_package = torch.load(encoder_path, map_location=lambda storage, loc: storage,weights_only=False)
        encoder_state_dict = encoder_package.get("model_state_dict", {})

    return model_state_dict, encoder_state_dict, args


# === Audio Inputs === #
class FireRedASRFeatureInputs(TensorSchema):
    type: Literal["audio_features"]
    input_features: Annotated[
        Union[torch.Tensor, list[torch.Tensor]],
        TensorShape("na", "nf", "nt"),
    ]

    feature_lengths: Annotated[
        torch.Tensor,
        TensorShape("na"),
    ]


class FireRedASREmbeddingInputs(TensorSchema):
    type: Literal["audio_embeds"] = "audio_embeds"

    audio_embeds: Annotated[
        list[torch.Tensor],
        TensorShape("bn", "naf", "hs"),
    ]


FireRedASRInputs = Union[FireRedASRFeatureInputs, FireRedASREmbeddingInputs]

# === Encoder Components === #
class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv2dSubsampling(nn.Module):
    def __init__(self, idim: int, d_model: int, out_channels: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, out_channels, 3, 2),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 2),
            nn.ReLU(),
        )
        subsample_idim = ((idim - 1) // 2 - 1) // 2
        self.out = nn.Linear(out_channels * subsample_idim, d_model)
        self.subsampling = 4
        left_context = right_context = 3
        self.context = left_context + 1 + right_context  # 7

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.conv(x)
        N, C, T, D = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(N, T, C * D))
        mask = x_mask[:, :, :-2:2][:, :, :-2:2]
        input_lengths = mask[:, -1, :].sum(dim=-1)
        return x, input_lengths, mask


class RelPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe_positive = torch.zeros(max_len, d_model, requires_grad=False)
        pe_negative = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(torch.log(torch.tensor(10000.0)).item()/d_model))
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Tmax = 2 * max_len - 1
        Tmax, T = self.pe.size(1), x.size(1)
        pos_emb = self.pe[:, Tmax // 2 - T + 1 : Tmax // 2 + T].clone().detach()
        return pos_emb



class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(0.0)
        self.INF = float('inf')

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        output, attn = self.forward_attention(attn, v, mask)
        return output, attn

    def forward_attention(self, attn, v, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
            mask = mask.eq(0)
            attn = attn.masked_fill(mask, -self.INF)
            attn = torch.softmax(attn, dim=-1).masked_fill(mask, 0.0)
        else:
            attn = torch.softmax(attn, dim=-1)

        d_attn = self.dropout(attn)
        output = torch.matmul(d_attn, v)

        return output, attn


class EncoderMultiHeadAttention(nn.Module):
    def __init__(self, n_head: int, d_model: int, residual_dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.d_v = self.d_k
        self.w_qs = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * self.d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * self.d_v, bias=False)
        self.layer_norm_q = nn.LayerNorm(d_model)
        self.layer_norm_k = nn.LayerNorm(d_model)
        self.layer_norm_v = nn.LayerNorm(d_model)
        self.attention = ScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.fc = nn.Linear(n_head * self.d_v, d_model, bias=False)
        self.dropout = nn.Dropout(residual_dropout)

    def forward(self, q, k, v, mask=None):
        sz_b, len_q = q.size(0), q.size(1)
        residual = q
        q, k, v = self.forward_qkv(q, k, v)
        output, attn = self.attention(q, k, v, mask=mask)
        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn

    def forward_qkv(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q = q.size(0), q.size(1)
        q = self.layer_norm_q(q)
        k = self.layer_norm_k(k)
        v = self.layer_norm_v(v)
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k).transpose(1, 2)
        k = self.w_ks(k).view(sz_b, k.size(1), n_head, d_k).transpose(1, 2)
        v = self.w_vs(v).view(sz_b, v.size(1), n_head, self.d_v).transpose(1, 2)
        return q, k, v

    def forward_output(self, output, residual, sz_b, len_q):
        output = output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        fc_out = self.fc(output)
        output = self.dropout(fc_out)
        output = output + residual
        return output


class RelPosMultiHeadAttention(EncoderMultiHeadAttention):
    def __init__(self, n_head, d_model,
                 residual_dropout=0.1):
        super().__init__(n_head, d_model,
                         residual_dropout)
        d_k = d_model // n_head
        self.scale = 1.0 / (d_k ** 0.5)
        self.linear_pos = nn.Linear(d_model, n_head * d_k, bias=False)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(n_head, d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(n_head, d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def _rel_shift(self, x):
        N, H, T1, T2 = x.size()
        zero_pad = torch.zeros((N, H, T1, 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(N, H, T2 + 1, T1)
        x = x_padded[:, :, 1:].view_as(x)
        x = x[:, :, :, : x.size(-1) // 2 + 1]
        return x

    def forward(self, q, k, v, pos_emb, mask=None):
        sz_b, len_q = q.size(0), q.size(1)

        residual = q
        q, k, v = self.forward_qkv(q, k, v)

        q = q.transpose(1, 2)
        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.n_head, self.d_k)
        p = p.transpose(1, 2)

        q_with_bias_u = (q + self.pos_bias_u.to(q.device)).transpose(1, 2)
        q_with_bias_v = (q + self.pos_bias_v.to(q.device)).transpose(1, 2)

        matrix_ac = torch.matmul(q_with_bias_u.to(k.dtype), k.transpose(-2, -1))

        matrix_bd = torch.matmul(q_with_bias_v.to(p.dtype), p.transpose(-2, -1))
        matrix_bd = self._rel_shift(matrix_bd)

        attn_scores = matrix_ac + matrix_bd
        attn_scores.mul_(self.scale)

        output, attn = self.attention.forward_attention(attn_scores, v, mask=mask)

        output = self.forward_output(output, residual, sz_b, len_q)
        return output, attn



class ConformerConvolution(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 33, dropout_rate: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1
        self.pre_layer_norm = nn.LayerNorm(d_model)
        self.pointwise_conv1 = nn.Conv1d(d_model, d_model * 4, kernel_size=1, bias=False)
        self.glu = nn.functional.glu
        self.padding = (kernel_size - 1) // 2
        self.depthwise_conv = nn.Conv1d(
            d_model * 2, d_model * 2, kernel_size, stride=1, padding=self.padding, groups=d_model * 2, bias=False
        )
        self.batch_norm = nn.LayerNorm(d_model * 2)
        self.swish = Swish()
        self.pointwise_conv2 = nn.Conv1d(d_model * 2, d_model, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        residual = x
        out = self.pre_layer_norm(x).transpose(1, 2)
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = self.pointwise_conv1(out)
        out = self.glu(out, dim=1)
        out = self.depthwise_conv(out)
        out = out.transpose(1, 2)
        out = self.swish(self.batch_norm(out))
        out = out.transpose(1, 2)
        out = self.dropout(self.pointwise_conv2(out))
        if mask is not None:
            out.masked_fill_(mask.ne(1), 0.0)
        out = out.transpose(1, 2)
        return out + residual


class ConformerFeedForward(nn.Module):
    def __init__(self, d_model: int, dropout_post: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            Swish(),
            nn.Dropout(dropout_post),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout_post),
        )

    def forward(self, x):
        residual = x
        output = self.net(x)
        return output + residual


class RelPosEmbConformerBlock(nn.Module):
    def __init__(self, d_model, n_head,
                 residual_dropout=0.1,
                 dropout_rate=0.1, kernel_size=33):
        super().__init__()
        self.ffn1 = ConformerFeedForward(d_model, dropout_rate)
        self.mhsa = RelPosMultiHeadAttention(n_head, d_model,
                                             residual_dropout)
        self.conv = ConformerConvolution(d_model, kernel_size,
                                         dropout_rate)
        self.ffn2 = ConformerFeedForward(d_model, dropout_rate)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, slf_attn_mask=None, pad_mask=None):
        out = 0.5 * x + 0.5 * self.ffn1(x)
        out = self.mhsa(out, out, out, pos_emb, mask=slf_attn_mask)[0]
        out = self.conv(out, pad_mask)
        out = 0.5 * out + 0.5 * self.ffn2(out)
        out = self.layer_norm(out)
        return out



class ConformerEncoder(nn.Module):
    def __init__(self, idim, n_layers, n_head, d_model,
                 residual_dropout=0.1, dropout_rate=0.1, kernel_size=33,
                 pe_maxlen=5000):
        super().__init__()
        self.odim = d_model

        self.input_preprocessor = Conv2dSubsampling(idim, d_model)
        self.positional_encoding = RelPositionalEncoding(d_model)
        self.dropout = nn.Dropout(residual_dropout)

        self.layer_stack = nn.ModuleList()
        for l in range(n_layers):
            block = RelPosEmbConformerBlock(d_model, n_head,
                        residual_dropout,
                        dropout_rate, kernel_size)
            self.layer_stack.append(block)

    def forward(self, padded_input, input_lengths, pad=True):
        if pad:
            padded_input = F.pad(padded_input,
                (0, 0, 0, self.input_preprocessor.context - 1), 'constant', 0.0)
        src_mask = self.padding_position_is_0(padded_input, input_lengths)

        embed_output, input_lengths, src_mask = self.input_preprocessor(padded_input, src_mask)
        enc_output = self.dropout(embed_output)

        pos_emb = self.dropout(self.positional_encoding(embed_output))

        enc_outputs = []
        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, pos_emb, slf_attn_mask=src_mask,
                                   pad_mask=src_mask)
            enc_outputs.append(enc_output)

        return enc_output, input_lengths, src_mask

    def padding_position_is_0(self, padded_input, input_lengths):
        N, T = padded_input.size()[:2]
        mask = torch.ones((N, T)).to(padded_input.device)
        for i in range(N):
            mask[i, input_lengths[i]:] = 0
        mask = mask.unsqueeze(dim=1)
        return mask.to(torch.uint8)


class FireRedASRAdapter(nn.Module):
    """Adapter module to project encoder outputs to LLM input space."""

    def __init__(self, encoder_dim: int, llm_dim: int, downsample_rate: int = 2):
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x: torch.Tensor, x_lens: torch.Tensor):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        x = x.contiguous()
        x = x.view(
            batch_size, seq_len // self.ds, feat_dim * self.ds
        )

        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens


# === FireRedASR Multimodal Processing Components === #

class FireRedASRProcessingInfo(BaseProcessingInfo):
    """Processing info for FireRedASR model."""

    def get_supported_mm_limits(self) -> Mapping[str, Optional[int]]:
        return {"audio": None}  # No limit on number of audio inputs

    def get_hf_processor(self, **kwargs: object):
        """FireRedASR uses custom audio processing, not HF processor."""
        # For FireRedASR, we use our own audio feature extraction
        return None

    # 增加一个可配置的下采样率，供处理器使用，避免访问模型实例
    @property
    def audio_downsample_rate(self) -> int:
        # 优先使用 runtime 覆盖值；其次从 HF 配置读取；最后默认 4
        try:
            if hasattr(self, "_audio_downsample_rate"):
                return int(getattr(self, "_audio_downsample_rate"))
        except Exception:
            pass
        try:
            cfg = getattr(self.model_config, 'hf_config', None)
            ds = getattr(cfg, 'encoder_downsample_rate', None) if cfg is not None else None
            if ds is not None:
                return int(ds)
        except Exception:
            pass
        return 4 

    def set_audio_downsample_rate(self, ds: int) -> None:
        self._audio_downsample_rate = int(ds)


class FireRedASRDummyInputsBuilder(BaseDummyInputsBuilder[FireRedASRProcessingInfo]):
    """Dummy inputs builder for FireRedASR model profiling."""

    def get_dummy_text(self, mm_counts: Mapping[str, int]) -> str:
        num_audios = mm_counts.get("audio", 0)
        # FireRedASR uses <|AUDIO|> as the default speech token
        if num_audios == 0:
            return ""

        # Use FireRedASR's speech token
        return "<speech>" * num_audios

    def get_dummy_mm_data(
        self,
        seq_len: int,
        mm_counts: Mapping[str, int],
        mm_options: Mapping[str, BaseDummyOptions] | None = None,
    ) -> MultiModalDataDict:
        num_audios = mm_counts.get("audio", 0)

        if num_audios == 0:
            return {}

        # Create dummy audio features - Use a reasonable size for profiling
        # FireRedASR typically processes 6-10 seconds of audio
        audio_len = 16000 * 6  # 6 seconds at 16kHz for dummy audio
        audio_overrides = mm_options.get("audio") if mm_options else None

        return {
            "audio": self._get_dummy_audios(length=audio_len, num_audios=num_audios, overrides=audio_overrides)
        }
def _fireredasr_field_config(hf_inputs: Mapping[str, torch.Tensor]):
    """Field configuration for FireRedASR inputs."""
    return dict(
        audio_embeds=MultiModalFieldConfig.batched("audio"),
        input_features=MultiModalFieldConfig.batched("audio"),
        feature_lengths=MultiModalFieldConfig.batched("audio"),
        audio_token_lengths=MultiModalFieldConfig.batched("audio"),
    )


class FireRedASRMultiModalDataParser(MultiModalDataParser):
    """Data parser for FireRedASR multimodal inputs."""

    def _parse_audio_data(
        self,
        data: Union[dict[str, torch.Tensor], ModalityData[AudioItem]],
    ) -> Optional[ModalityDataItems[Any, Any]]:
        if isinstance(data, dict):
            return DictEmbeddingItems(
                data,
                modality="audio",
                required_fields={"audio_embeds"},
                fields_factory=_fireredasr_field_config,
            )

        return super()._parse_audio_data(data)


class FireRedASRMultiModalProcessor(BaseMultiModalProcessor[FireRedASRProcessingInfo]):
    """Multimodal processor for FireRedASR model (encoder-decoder interface)."""

    def __init__(
        self,
        info: FireRedASRProcessingInfo,
        dummy_inputs: BaseDummyInputsBuilder[FireRedASRProcessingInfo],
        *,
        cache=None,
    ) -> None:
        super().__init__(info, dummy_inputs, cache=cache)
        # Ensure decoder_start_token_id is configured on HF config before
        # InputPreprocessor queries it. This runs early when the processor
        # is created by InputPreprocessor.
        try:
            hf_cfg = self.info.model_config.hf_config
        except Exception:
            hf_cfg = None
        if hf_cfg is not None and getattr(hf_cfg, 'decoder_start_token_id', None) is None:
            # Avoid depending on tokenizer initialization. Use a stable default.
            try:
                setattr(hf_cfg, 'decoder_start_token_id', 1)
            except Exception:
                pass
        # Initialize CMVN stats to align audio preprocessing with FireRedASR
        # reference implementation (ASRFeatExtractor).
        self._cmvn_means = None  # type: Optional[torch.Tensor]
        self._cmvn_istd = None   # type: Optional[torch.Tensor]
        try:
            self._init_cmvn_stats()
        except Exception:
            # Proceed without CMVN if initialization fails.
            self._cmvn_means = None
            self._cmvn_istd = None
    
    def _get_dummy_inputs_builder(self) -> BaseDummyInputsBuilder:
        """Returns the custom dummy inputs builder for FireRedASR."""
        return FireRedASRDummyInputsBuilder(self.info)

    def _get_data_parser(self) -> MultiModalDataParser:
        # FireRedASR expects 16kHz audio input
        return FireRedASRMultiModalDataParser(target_sr=16000)

    @property
    def pad_dummy_encoder_prompt(self) -> bool:
        # For profiling, pad dummy encoder prompt to match audio token count
        return True

    def _calculate_accurate_token_lengths(self, feature_lengths: torch.Tensor):

        batch_size = feature_lengths.shape[0]
        token_lengths = []

        for i in range(batch_size):
            orig_len = int(feature_lengths[i].item())

            padded_len = orig_len + 6  # context-1 = 6

            after_conv1 = max(1, (padded_len - 3) // 2 + 1)

            after_conv2 = max(1, (after_conv1 - 3) // 2 + 1)
            try:
                ds_rate = self.info.audio_downsample_rate//2
            except:
                ds_rate = 2  # 默认值

            usable_len = after_conv2 - (after_conv2 % ds_rate)
            final_len = max(1, usable_len // ds_rate)

            token_lengths.append(final_len)

        return token_lengths


    def _call_hf_processor(
        self,
        prompt: str,
        mm_data: Mapping[str, object],
        mm_kwargs: Mapping[str, Any],
        tok_kwargs: Mapping[str, object],
    ) -> BatchFeature:
        """Process audio data for FireRedASR."""
        from transformers import BatchFeature
        import numpy as np
        # Prefer torchaudio Kaldi fbank; fall back to librosa
        import torchaudio  # type: ignore
        import kaldi_native_fbank as knf  # type: ignore
        import kaldiio  # type: ignore

        audios = mm_data.pop("audios", [])
        if audios:
            mm_data["audio"] = audios
        # For text-only input
        if not mm_data.get("audio", []):
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # Process audio inputs
        audios = mm_data.get("audio", [])
        if not isinstance(audios, list):
            audios = [audios]

        # Validate that we have actual audio data
        if not audios or len(audios) == 0:
            prompt_ids = self.info.get_tokenizer().encode(prompt)
            prompt_ids = self._apply_hf_processor_tokens_only(prompt_ids)
            return BatchFeature(dict(input_ids=[prompt_ids]), tensor_type="pt")

        # Extract Fbank features for each audio
        input_features_list = []
        feature_lengths_list = []
        audios = [audio *  32768 for audio in audios]
        for audio_item in audios:

            if isinstance(audio_item, tuple):
                audio_data, sample_rate = audio_item
            else:
                audio_data = audio_item
                sample_rate = 16000  # Default sample rate

            # Convert to numpy mono float32 waveform
            if isinstance(audio_data, torch.Tensor):
                audio_data = audio_data.detach().cpu().numpy()

            # Compute fbank features — prefer kaldi-native-fbank to match HF
            feats: torch.Tensor
            import kaldi_native_fbank as knf  # type: ignore
            opts = knf.FbankOptions()
            opts.frame_opts.dither = 0.0
            opts.mel_opts.num_bins = 80
            opts.frame_opts.snip_edges = True
            opts.mel_opts.debug_mel = False
            fbank = knf.OnlineFbank(opts)
            fbank.accept_waveform(sample_rate, audio_data.tolist())
            frames =[]
            for i in range(fbank.num_frames_ready):
                frames.append(fbank.get_frame(i))
            if len(frames) == 0:
                feats = torch.zeros((0, opts.mel_opts.num_bins), dtype=torch.float32)
            feats_np = np.vstack(frames)
            feats = torch.from_numpy(feats_np).float()

            # Apply CMVN — global when available, else per-utterance
            try:
                feats = (feats - self._cmvn_means.to(feats.device, feats.dtype)) * self._cmvn_istd.to(feats.device, feats.dtype)
            except Exception:
                mean = feats.mean(dim=0, keepdim=True)
                std = feats.std(dim=0, keepdim=True)
                std = torch.where(std < 1e-6, torch.full_like(std, 1.0), std)
                feats = (feats - mean) / std

            input_features_list.append(feats)
            feature_lengths_list.append(int(feats.shape[0]))
        # Safety check: ensure we have features
        if not input_features_list or len(input_features_list) == 0:
            # Create dummy features if no valid audio could be processed
            input_features_list = [torch.zeros(100, 80, dtype=torch.float32)]  # 100 time steps, 80 mel bins
            feature_lengths_list = [100]

        # Pad features to same length
        max_len = max(feat.shape[0] for feat in input_features_list)
        if max_len == 0:
            max_len = 100  # Minimum fallback

        padded_features = []
        for features in input_features_list:
            if features.shape[0] < max_len:
                # Ensure padding has the same dtype as the features
                padding = torch.zeros(max_len - features.shape[0], features.shape[1],
                                    dtype=features.dtype, device=features.device)
                features = torch.cat([features, padding], dim=0)
            padded_features.append(features)

        # Stack into batch tensor (B, T, F) then transpose to (B, F, T)
        input_features = torch.stack(padded_features, dim=0).transpose(1, 2)  # (B, F, T)
        feature_lengths = torch.tensor(feature_lengths_list, dtype=torch.long)
        tokenizer = self.info.get_tokenizer()

        speech_token_id = tokenizer.convert_tokens_to_ids("<speech>")
        if speech_token_id == getattr(tokenizer, 'unk_token_id', None) or speech_token_id is None:
            try:
                special_tokens_dict = {"additional_special_tokens": ["<speech>"]}
                tokenizer.add_special_tokens(special_tokens_dict)
                speech_token_id = tokenizer.convert_tokens_to_ids("<speech>")
            except Exception:
                speech_token_id = 151646  # Fallback

        # Create message following FireRedASR template
        messages = [
            {"role": "user", "content": "<speech>请转写音频为文字"},
            {"role": "assistant", "content": ""}
        ]
        # Use FireRedASR's decode template for inference
        TEMPLATE = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\\n' + message['content']}}{% if loop.last %}{{''}}{% else %}{{ '<|im_end|>\\n' }}{% endif %}{% endfor %}"
        # Generate the template tokens
        template_tokens= tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            chat_template=TEMPLATE,
            add_generation_prompt=False,
            padding=False,
        )
        
        audio_token_lengths = self._calculate_accurate_token_lengths(feature_lengths)
        num_audio_tokens = audio_token_lengths[0]
        index_of = template_tokens.index(speech_token_id) + 1
        template_tokens[index_of:index_of] = [speech_token_id] * num_audio_tokens
        prompt_ids = template_tokens
        audio_token_lengths = torch.tensor([num_audio_tokens], dtype=torch.long)

        return BatchFeature(
            dict(
                input_ids=[prompt_ids],
                input_features=input_features,
                feature_lengths=feature_lengths,
                audio_token_lengths=audio_token_lengths,
            ),
            tensor_type="pt"
        )

    def _init_cmvn_stats(self) -> None:
        """Initialize CMVN stats from env or HF config.

        Tries the following in order:
        - Env var `FIREREDASR_CMVN_PATH`
        - HF config field `cmvn_path`
        - If model path is a directory, `<model>/cmvn.ark`
        """
        import os
        cmvn_path = os.getenv('FIREREDASR_CMVN_PATH', '/workspace/bella-infra/user/zhangshuge002/FireRedASR/FireRedASR/out/fireredasr-llm-hf-test/cmvn.ark')
        if not cmvn_path:
            try:
                cfg = getattr(self.info.model_config, 'hf_config', None)
                cmvn_path = getattr(cfg, 'cmvn_path', None) if cfg is not None else None
            except Exception:
                cmvn_path = None
        if not cmvn_path:
            # Try to resolve from model directory
            try:
                model_dir = getattr(self.info.model_config, 'model', None)
                if isinstance(model_dir, str) and os.path.isdir(model_dir):
                    candidate = os.path.join(model_dir, 'cmvn.ark')
                    if os.path.exists(candidate):
                        cmvn_path = candidate
            except Exception:
                pass
        if not cmvn_path or not os.path.exists(cmvn_path):
            # CMVN not available
            self._cmvn_means = None
            self._cmvn_istd = None
            return

        # Load Kaldi CMVN stats
        try:
            import kaldiio  # type: ignore
            stats = kaldiio.load_mat(cmvn_path)
        except Exception:
            # If kaldiio is not available, skip CMVN
            self._cmvn_means = None
            self._cmvn_istd = None
            return

        # stats shape: (2, dim+1)
        dim = stats.shape[-1] - 1
        count = float(stats[0, dim])
        if count < 1.0:
            self._cmvn_means = None
            self._cmvn_istd = None
            return
        floor = 1e-20
        means =[]
        inv_std =[]
        for d in range(dim):
            mean = float(stats[0, d]) / count
            var = float(stats[1, d]) / count - mean * mean
            if var < floor:
                var = floor
            istd = 1.0 / (var ** 0.5)
            means.append(mean)
            inv_std.append(istd)
        # Store as tensors of shape (1, dim) to broadcast over time
        self._cmvn_means = torch.tensor(means, dtype=torch.float32).unsqueeze(0)
        self._cmvn_istd = torch.tensor(inv_std, dtype=torch.float32).unsqueeze(0)

    def _get_mm_fields_config(
        self,
        hf_inputs: BatchFeature,
        hf_processor_mm_kwargs: Mapping[str, object],
    ) -> Mapping[str, MultiModalFieldConfig]:
        return _fireredasr_field_config(hf_inputs)

    def _get_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        hf_processor_mm_kwargs: Mapping[str, object],
        out_mm_kwargs: MultiModalKwargsItems,
    ) -> Sequence[PromptUpdate]:
        """Generate prompt updates for FireRedASR."""

        out_mm_data = out_mm_kwargs.get_data()
        tokenizer = self.info.get_tokenizer()
        speech_token_id = tokenizer.convert_tokens_to_ids("<speech>")
        def get_replacement_fireredasr(item_idx: int):
            # Get tokenizer for template processing
            # Calculate number of audio tokens for this item
            if "audio_token_lengths" in out_mm_data:
                atl = out_mm_data["audio_token_lengths"]
                try:
                    if isinstance(atl, torch.Tensor):
                        value = int(atl[int(item_idx)].item())
                    elif isinstance(atl, (list, tuple)):
                        value = int(atl[int(item_idx)])
                    else:
                        import numpy as np
                        if isinstance(atl, np.ndarray):
                            value = int(atl[int(item_idx)])
                        else:
                            value = 1
                except Exception:
                    value = 1
                num_features = max(1, value)

                final_tokens = [speech_token_id] * (num_features )
                # final_tokens = template_tokens

                return PromptUpdateDetails.select_token_id(
                    final_tokens,
                    embed_token_id=speech_token_id,
                )
        return [
            PromptReplacement(
                modality="audio",
                target="<speech>",  # 使用列表形式的token ID
                replacement=get_replacement_fireredasr,
            )
        ]
    


@MULTIMODAL_REGISTRY.register_processor(
    FireRedASRMultiModalProcessor,
    info=FireRedASRProcessingInfo,
    dummy_inputs=FireRedASRDummyInputsBuilder)
class FireRedASRForSpeechToText(nn.Module, SupportsTranscription, SupportsMultiModal, SupportsPP):
    """FireRedASR LLM model for Speech-to-Text transcription."""

    # Required by SupportsTranscription interface
    supported_languages = {
        "en": "english",
        "zh": "chinese",
        "ja": "japanese",
        "ko": "korean",
        # Add more languages as needed
    }
    supports_transcription = True
    supports_transcription_only = True  # Pure ASR model

    @classmethod
    def get_placeholder_str(cls, modality: str, i: int) -> Optional[str]:
        """Get placeholder string for audio input."""
        if modality.startswith("audio"):
            # 使用参考实现的语音占位符
            return "<speech>"
        return None

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config
        multimodal_config = vllm_config.model_config.multimodal_config

        self.speech_lens: torch.Tensor

        self.config = config
        self.test = False
        # Ensure vLLM treats this model as encoder-decoder for V1 pipeline
        # so that the encoder flow runs and get_multimodal_embeddings is used.
        # try:
        #     setattr(self.config, 'is_encoder_decoder', True)
        # except Exception:
        #     pass
        self.multimodal_config = multimodal_config

        # FireRedASR components
        # FireRedASR encoder hidden dim：优先使用 d_model，其次 encoder_dim
        llm_dim = getattr(config, 'hidden_size', 3584)  # Qwen2-7B hidden size
        downsample_rate = getattr(config, 'encoder_downsample_rate', 2)

        # Speech encoder (Conformer) — 参考实现参数命名
        self.speech_encoder = ConformerEncoder(
            idim=getattr(config, 'idim', getattr(config, 'encoder_input_size', 80)),
            n_head=20,
            d_model=1280,
            n_layers = 16,
        )

        # Adapter/Projector
        self.multi_modal_projector = FireRedASRAdapter(
            encoder_dim=1280, llm_dim=llm_dim, downsample_rate=downsample_rate
        )

        self.quant_config = quant_config

        # Language model (Qwen2) - create a compatible config for the LLM part

        llm_config = Qwen2Config(
            vocab_size=getattr(config, 'vocab_size', 152064),
            hidden_size=llm_dim,
            intermediate_size=getattr(config, 'intermediate_size', 18944),
            num_hidden_layers=getattr(config, 'num_hidden_layers', 28),
            num_attention_heads=getattr(config, 'num_attention_heads', 28),
            num_key_value_heads=getattr(config, 'num_key_value_heads', 4),
            hidden_act=getattr(config, 'hidden_act', 'silu'),
            max_position_embeddings=getattr(config, 'max_position_embeddings', 32768),
            initializer_range=getattr(config, 'initializer_range', 0.02),
            rms_norm_eps=getattr(config, 'rms_norm_eps', 1e-6),
            use_cache=getattr(config, 'use_cache', True),
            tie_word_embeddings=getattr(config, 'tie_word_embeddings', False),
            rope_theta=getattr(config, 'rope_theta', 1000000.0),
            use_sliding_window=getattr(config, 'use_sliding_window', False),
            sliding_window=getattr(config, 'sliding_window', 131072),
            max_window_layers=getattr(config, 'max_window_layers', 28),
            attention_dropout=getattr(config, 'attention_dropout', 0.0),
            bos_token_id=getattr(config, 'bos_token_id', 151643),
            eos_token_id=getattr(config, 'eos_token_id', 151645),
            model_type=getattr(config, 'model_type', 'qwen2'),
            
        )

        self.language_model = init_vllm_registered_model(
            vllm_config=vllm_config,
            hf_config=llm_config,
            prefix=maybe_prefix(prefix, "language_model"),
            architectures=["Qwen2ForCausalLM"],
        )

        # Special token IDs and tokenizer for ASR
        self.speech_token_id = getattr(config, 'speech_token_id', None)
        self.tokenizer = self._get_tokenizer(vllm_config)

        # Ensure decoder start token id is set for encoder-decoder preprocessing.
        # If not provided by the HF config, fall back to tokenizer BOS/EOS or 1.
        try:
            dec_start = getattr(config, 'decoder_start_token_id', None)
        except Exception:
            dec_start = None
        if dec_start is None:
            bos_id = getattr(self.tokenizer, 'bos_token_id', None)
            eos_id = getattr(self.tokenizer, 'eos_token_id', None)
            if isinstance(bos_id, int):
                dec_start = bos_id
            elif isinstance(eos_id, int):
                dec_start = eos_id
            else:
                dec_start = 1  # safe fallback
            try:
                setattr(config, 'decoder_start_token_id', dec_start)
            except Exception:
                pass

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

        # Load FireRedASR weights if checkpoint paths are provided
        self._load_fireredasr_checkpoints(config, vllm_config)

    def _load_fireredasr_checkpoints(self, config, vllm_config: VllmConfig):
        """Load FireRedASR model weights from checkpoint files."""
        # Check for checkpoint paths in config
        model_path = getattr(config, 'fireredasr_model_path', None)
        encoder_path = getattr(config, 'fireredasr_encoder_path', None)

        # Also check in model config for alternative path locations
        model_config = vllm_config.model_config
        model_dir = getattr(model_config, 'model', '')

        # Try to find checkpoint files in the model directory
        if model_path is None and model_dir:
            candidate_model_path = os.path.join(model_dir, 'model.pth.tar')
            if os.path.exists(candidate_model_path):
                model_path = candidate_model_path

        if encoder_path is None and model_dir:
            candidate_encoder_path = os.path.join(model_dir, 'asr_encoder.pth.tar')
            if os.path.exists(candidate_encoder_path):
                encoder_path = candidate_encoder_path

        # Load weights if checkpoint files are found
        if model_path and os.path.exists(model_path):
            try:
                model_state_dict, encoder_state_dict, args = _load_firered_llm_weights(
                    model_path, encoder_path
                )

                # CORRECTED LOGIC based on FireRedASR implementation:
                # 1. Encoder weights: Load from asr_encoder.pth.tar (AED model's encoder)
                # 2. Adapter weights: Load from model.pth.tar with 'encoder_projector.' prefix

                # FINAL CORRECTED LOGIC:
                # Based on the actual checkpoint content you provided:
                # - model.pth.tar contains encoder weights with 'encoder.' prefix
                # - asr_encoder.pth.tar only contains 'args', no actual weights
                # So we should load encoder weights from model.pth.tar, not asr_encoder.pth.tar

                # Load encoder weights from model.pth.tar (where they actually are!)
                if model_state_dict and hasattr(self, 'speech_encoder'):
                    # Get expected parameter names AND buffer names from our speech_encoder
                    expected_params = set(name for name, _ in self.speech_encoder.named_parameters())
                    expected_buffers = set(name for name, _ in self.speech_encoder.named_buffers())
                    expected_all = expected_params.union(expected_buffers)

                    encoder_weights = {}

                    # Extract encoder weights with 'encoder.' prefix from model.pth.tar
                    for key, value in model_state_dict.items():
                        if key.startswith('encoder.'):
                            new_key = key[8:]  # Remove 'encoder.' prefix

                            # Check if this parameter/buffer exists in our model
                            if new_key in expected_all:
                                encoder_weights[new_key] = value

                    if encoder_weights:
                        missing_keys, unexpected_keys = self.speech_encoder.load_state_dict(
                            encoder_weights, strict=False
                        )
                        if missing_keys:
                            print(f"Warning: {len(missing_keys)} encoder keys not found in checkpoint")
                        if unexpected_keys:
                            print(f"Warning: {len(unexpected_keys)} unexpected encoder keys in checkpoint")

                # Load adapter/projector weights from model.pth.tar
                if model_state_dict and hasattr(self, 'multi_modal_projector'):
                    # Get expected parameter names from our multi_modal_projector
                    expected_params = set(name for name, _ in self.multi_modal_projector.named_parameters())

                    adapter_weights = {}

                    # In model.pth.tar, adapter weights have 'encoder_projector.' prefix
                    for key, value in model_state_dict.items():
                        if key.startswith('encoder_projector.'):
                            new_key = key[18:]  # len('encoder_projector.') = 18
                            if new_key in expected_params:
                                adapter_weights[new_key] = value

                    if adapter_weights:
                        missing_keys, unexpected_keys = self.multi_modal_projector.load_state_dict(
                            adapter_weights, strict=False
                        )
                        if missing_keys:
                            print(f"Warning: {len(missing_keys)} adapter keys not found in checkpoint")
                        if unexpected_keys:
                            print(f"Warning: {len(unexpected_keys)} unexpected adapter keys in checkpoint")

                # Verify weights were loaded properly
                weights_ok = self._verify_weights_loaded()
                if weights_ok:
                    print(f"Successfully loaded FireRedASR checkpoints from {model_path}")
                    if encoder_path:
                        print(f"Encoder weights loaded from {encoder_path}")
                else:
                    print(f"Warning: Some weights may not have loaded correctly from {model_path}")

            except Exception as e:
                print(f"Warning: Failed to load FireRedASR checkpoints: {e}")
                print("Continuing with randomly initialized weights...")
        else:
            print("No FireRedASR checkpoint files found. Using randomly initialized weights.")

    def _verify_weights_loaded(self):
        """Verify that weights were actually loaded by checking parameter statistics."""
        # Check encoder weights
        encoder_loaded = 0
        encoder_total = 0
        for name, param in self.speech_encoder.named_parameters():
            encoder_total += 1
            if param.abs().max() > 1e-6:  # Has reasonable non-zero values
                encoder_loaded += 1

        # Check adapter weights
        adapter_loaded = 0
        adapter_total = 0
        for name, param in self.multi_modal_projector.named_parameters():
            adapter_total += 1
            if param.abs().max() > 1e-6:
                adapter_loaded += 1

        # Report results
        total_loaded = encoder_loaded + adapter_loaded
        total_params = encoder_total + adapter_total
        success_rate = (total_loaded / total_params) * 100 if total_params > 0 else 0

        if success_rate < 90:  # Only warn if success rate is low
            print(f"Warning: Only {success_rate:.1f}% of parameters appear to be loaded correctly")

        return success_rate > 90

    def _get_tokenizer(self, vllm_config: VllmConfig):
        """Get the tokenizer for text generation."""
        from transformers import AutoTokenizer
        import os
        env_tok = os.getenv('FIREREDASR_TOKENIZER_PATH', None)
        cfg = getattr(vllm_config.model_config, 'hf_config', None)
        cfg_tok = getattr(cfg, 'fireredasr_tokenizer_path', None) if cfg is not None else None
        user_default = "/workspace/bella-infra/user/libeibei031/vllm_FrieRedASR/FireRedASR/tools/out/fire-red-asr-aed-hf"
        model_path = getattr(vllm_config.model_config, 'model', '')
        tok_path = env_tok or cfg_tok or (model_path if model_path else user_default)
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
        except Exception:
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except Exception:
                tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B-Instruct")
        # Ensure DEFAULT_SPEECH_TOKEN exists
        try:
            tokenizer.add_special_tokens({"additional_special_tokens": ["<speech>"]})
        except Exception:
            pass
        # Align pad/bos/eos to reference if available
        try:
            pad_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
            if isinstance(pad_id, int) and pad_id >= 0:
                tokenizer.pad_token = "<|endoftext|>"
                tokenizer.pad_token_id = pad_id
        except Exception:
            pass
        try:
            bos_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            if isinstance(bos_id, int) and bos_id >= 0:
                tokenizer.bos_token = "<|im_start|>"
                tokenizer.bos_token_id = bos_id
        except Exception:
            pass
        try:
            eos_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if isinstance(eos_id, int) and eos_id >= 0:
                tokenizer.eos_token = "<|im_end|>"
                tokenizer.eos_token_id = eos_id
        except Exception:
            pass
        # Cache speech token id
        try:
            self.speech_token_id = tokenizer.convert_tokens_to_ids("<speech>")
        except Exception:
            pass
        return tokenizer

    def _validate_and_reshape_mm_tensor(self, mm_input: object,
                                       name: str) -> torch.Tensor | list[torch.Tensor]:
        """Validate and normalize multimodal tensors to expected shapes.

        - input_features: expect (B, F, T) torch.Tensor. If a list of (T, F),
          pad to max T, stack to (B, T, F) then transpose to (B, F, T).
        - feature_lengths: return 1D long tensor of shape (B,).
        - audio_embeds: keep as list of tensors.
        """
        # audio_embeds are passed through as list for later tuple conversion
        if name == "audio_embeds":
            if isinstance(mm_input, (list, tuple)):
                return [x if isinstance(x, torch.Tensor) else torch.as_tensor(x)
                        for x in mm_input]
            if isinstance(mm_input, torch.Tensor):
                return [mm_input]
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")

        if name == "feature_lengths":
            if isinstance(mm_input, torch.Tensor):
                t = mm_input.to(dtype=torch.long)
                # Expect 1D (B,). If given (B, 1) 或更高维，拉平成 1D。
                if t.ndim == 1:
                    return t
                return t.reshape(-1)
            if isinstance(mm_input, (list, tuple)):
                return torch.tensor(list(mm_input), dtype=torch.long)
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")

        if name == "input_features":
            if isinstance(mm_input, torch.Tensor):
                # Expect (N, F, T). If higher rank, merge leading dims.
                if mm_input.ndim == 3:
                    return mm_input
                elif mm_input.ndim > 3:
                    # Flatten leading dimensions into a single N
                    F = mm_input.shape[-2]
                    T = mm_input.shape[-1]
                    return mm_input.reshape(-1, F, T).contiguous()
                else:
                    # If 2D, interpret as (T, F) then expand batch=1
                    if mm_input.ndim == 2:
                        return mm_input.transpose(0, 1).unsqueeze(0)
                    raise ValueError(f"input_features must be 2D/3D/4D, got {mm_input.ndim}D")
            if isinstance(mm_input, (list, tuple)):
                tensors: list[torch.Tensor] = [
                    x if isinstance(x, torch.Tensor) else torch.as_tensor(x) for x in mm_input
                ]
                # Ensure each is (T, F)
                proc: list[torch.Tensor] =[]
                max_len = 0
                for t in tensors:
                    if t.ndim == 2:
                        proc.append(t)
                    elif t.ndim == 3 and t.shape[0] == 1:
                        proc.append(t.squeeze(0))
                    else:
                        # Best effort: flatten to (T, F)
                        proc.append(t.reshape(t.shape[0], -1))
                    max_len = max(max_len, proc[-1].shape[0])

                padded: list[torch.Tensor] =[]
                for t in proc:
                    if t.shape[0] < max_len:
                        pad = torch.zeros(max_len - t.shape[0], t.shape[1], dtype=t.dtype, device=t.device)
                        t = torch.cat([t, pad], dim=0)
                    padded.append(t)
                # (B, T, F) -> (B, F, T)
                return torch.stack(padded, dim=0).transpose(1, 2).contiguous()
            raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")

        # Fallback: return tensor as-is
        if isinstance(mm_input, torch.Tensor):
            return mm_input
        raise ValueError(f"Incorrect type of {name}. Got type: {type(mm_input)}")

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[FireRedASRInputs]:
        input_features = kwargs.pop('input_features', None)
        audio_embeds = kwargs.pop('audio_embeds', None)
        feature_lengths = kwargs.pop('feature_lengths', None)

        if input_features is None and audio_embeds is None:
            return None

        if audio_embeds is not None:
            if not isinstance(audio_embeds, (torch.Tensor, list)):
                raise ValueError("Incorrect type of audio embeds."
                               f"Got type: {type(audio_embeds)}")
            audio_embeds = self._validate_and_reshape_mm_tensor(
                audio_embeds, "audio_embeds")
            return FireRedASREmbeddingInputs(type="audio_embeds",
                                           audio_embeds=audio_embeds)

        if input_features is not None:
            input_features = self._validate_and_reshape_mm_tensor(
                input_features, 'input_features')
            if feature_lengths is not None:
                feature_lengths = self._validate_and_reshape_mm_tensor(
                    feature_lengths, 'feature_lengths')
            else:
                # Create default lengths if not provided
                batch_size = input_features.shape[0]
                seq_len = input_features.shape[-1]
                feature_lengths = torch.full((batch_size,), seq_len,
                                           dtype=torch.long,
                                           device=input_features.device)

            return FireRedASRFeatureInputs(
                type="audio_features",
                input_features=input_features,
                feature_lengths=feature_lengths)

        raise AssertionError("This line should be unreachable.")

    def _process_audio_input(
        self, audio_input: FireRedASRInputs
    ) -> Union[torch.Tensor, tuple[torch.Tensor, ...]]:
        if audio_input["type"] == "audio_embeds":
            audio_embeds = audio_input["audio_embeds"]
            return tuple(audio_embeds)

        # Process raw audio features
        input_features = audio_input["input_features"]
        feature_lengths = audio_input["feature_lengths"]

        # Transpose from (B, F, T) to (B, T, F) for encoder
        input_features = input_features.transpose(1, 2)

        # Ensure input tensor matches the encoder dtype (for mixed precision)
        target_dtype = None
        try:
            target_dtype = self.speech_encoder.input_preprocessor.out.weight.dtype
        except Exception:
            # fallback to any parameter's dtype
            try:
                target_dtype = next(self.speech_encoder.parameters()).dtype
            except StopIteration:
                target_dtype = input_features.dtype
        if input_features.dtype != target_dtype:
            input_features = input_features.to(dtype=target_dtype)

        # Pass through speech encoder
        encoder_out, enc_lengths, enc_mask = self.speech_encoder(
            input_features, feature_lengths)

        # Pass through adapter/projector
        speech_features, speech_lens = self.multi_modal_projector(
            encoder_out, enc_lengths)

        self.speech_lens = speech_lens
        return speech_features

    def get_language_model(self) -> torch.nn.Module:
        return self.language_model

    def process_audio_features(self,
                                 **kwargs: object) -> torch.Tensor:
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []
        speech_features = self._process_audio_input(audio_input)
        return speech_features

    def get_multimodal_embeddings(self, **kwargs: object) -> MultiModalEmbeddings:
        """Get multimodal embeddings for audio input."""
        # This method is required by SupportsMultiModal interface
        audio_input = self._parse_and_validate_audio_input(**kwargs)
        if audio_input is None:
            return []

        speech_features = self._process_audio_input(audio_input)
        # Convert tuple of tensors to list for MultiModalEmbeddings
        if isinstance(speech_features, tuple):
            return list(speech_features)
        return speech_features

    def get_input_embeddings(
        self,
        input_ids: torch.Tensor,  # [B, seq_len]
        multimodal_embeddings: list[list[torch.Tensor]] | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        支持 batch 输入的多模态 embedding 获取函数
        multimodal_embeddings: 
            若提供，为长度为 B 的列表，每个元素是当前样本的语音特征列表 [speech_1, speech_2, ...]。
        """
        batch_size = len(multimodal_embeddings) 
        new_input_ids = self.remove_duplicate_token_batch(input_ids)  # 假定支持 batch
        ids_len = len(new_input_ids)
        b_idx = 0
        merged_embeds = []
        for idx_ids in range(ids_len) :
            now_input_ids = new_input_ids[idx_ids].unsqueeze(0)
            inputs_embeds = self.language_model.get_input_embeddings(now_input_ids)  # [B, seq_len, D]
            if (now_input_ids == 151646).any().item() and b_idx < batch_size:
                sample_multimodal = multimodal_embeddings[b_idx].unsqueeze(0)
                merged = self._merge_input_ids_with_speech_features(
                    speech_features=sample_multimodal,
                    inputs_embeds=inputs_embeds,
                    input_ids=now_input_ids,
                    speech_lens=self.speech_lens,
                )
                merged_embeds.append(merged.squeeze(0)) 
                b_idx = b_idx + 1
            else :
                merged_embeds.append(inputs_embeds.squeeze(0))

        merged_embeds = torch.cat(merged_embeds, dim=0)
        merged_embeds = torch.cat([merged_embeds, merged_embeds.new_zeros(input_ids.size(0) - merged_embeds.size(0), *merged_embeds.shape[1:])], dim=0)

        return  merged_embeds

    def _merge_input_ids_with_speech_features(
        self, speech_features, inputs_embeds, input_ids, labels=None,
        speech_lens=None
    ):
        """
        Modified from: https://github.com/k2-fsa/icefall/blob/master/egs/speech_llm/ASR_LLM/whisper_llm_zh/model.py
        """
        speech_lens = None
        placeholder_id = self._get_audio_placeholder_id()
        num_speechs, speech_len, embed_dim = speech_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(151643)
        )
        # 1. Create a mask to know where special speech tokens are
        special_speech_token_mask = input_ids == placeholder_id
        num_special_speech_tokens = torch.sum(special_speech_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (
            num_special_speech_tokens.max() * (speech_len - 1)
        ) + sequence_length
        batch_indices, non_speech_indices = torch.where(
            input_ids != placeholder_id
        )

        new_token_positions = (
            torch.cumsum((special_speech_token_mask * (speech_len - 1) + 1), -1) - 1
        )  # (N,U)
        nb_speech_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_speech_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_speech_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size,
            max_embed_dim,
            embed_dim,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        target_device = inputs_embeds.device
        batch_indices, non_speech_indices, text_to_overwrite = (
            batch_indices.to(target_device),
            non_speech_indices.to(target_device),
            text_to_overwrite.to(target_device),
        )
        # attention_mask = attention_mask.to(target_device)

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<speech>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the speech features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_speech_indices
        ]
        # final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
        #     batch_indices, non_speech_indices
        # ]
        # if labels is not None:
        #     final_labels[batch_indices, text_to_overwrite] = labels[
        #         batch_indices, non_speech_indices
        #     ]

        # 5. Fill the embeddings corresponding to the speechs. Anything that is not `text_positions` needs filling (#29835)
        speech_to_overwrite = torch.full(
            (batch_size, max_embed_dim),
            True,
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        speech_to_overwrite[batch_indices, text_to_overwrite] = False
        if speech_lens is not None:
            speech_pad_position = speech_to_overwrite.cumsum(-1) <= speech_lens[:, None]
        speech_to_overwrite &= speech_to_overwrite.cumsum(-1) - 1 >= nb_speech_pad[
            :, None
        ].to(target_device)

        if speech_to_overwrite.sum() != speech_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of speech tokens is {torch.sum(special_speech_token_mask)} while"
                f" the number of speech given to the model is {num_speechs}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[speech_to_overwrite] = (
            speech_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        if speech_lens is not None:
            speech_to_overwrite &= speech_pad_position
        # final_attention_mask |= speech_to_overwrite

        # 6. Mask out the embedding at padding positions, as we later use the past_key_value value to determine the non-attended tokens.
        batch_indices, pad_indices = torch.where(
            input_ids == 151643
        )
        indices_to_mask = new_token_positions[batch_indices, pad_indices]

        final_embedding[batch_indices, indices_to_mask] = 0

        # if labels is None:
        #     final_labels = None

        return final_embedding #, position_ids


    def _get_audio_placeholder_id(self) -> int:
        """Get the speech token ID following FireRedASR's approach."""
        # Prefer configured speech token id if available
        tok = getattr(self, 'tokenizer', None)
        placeholder_id = getattr(self, 'speech_token_id', None)
        unk_id = getattr(tok, 'unk_token_id', None) if tok is not None else None

        if isinstance(placeholder_id, int) and placeholder_id != unk_id:
            return int(placeholder_id)

        if tok is not None:
            # FireRedASR uses "<speech>" as the primary speech token
            try:
                speech_id = tok.convert_tokens_to_ids("<speech>")
                if isinstance(speech_id, int) and speech_id != unk_id:
                    return int(speech_id)
            except Exception:
                pass

            # If <speech> token doesn't exist, try to add it dynamically
            try:
                special_tokens_dict = {"additional_special_tokens": ["<speech>"]}
                tok.add_special_tokens(special_tokens_dict)
                speech_id = tok.convert_tokens_to_ids("<speech>")
                if isinstance(speech_id, int) and speech_id != unk_id:
                    return int(speech_id)
            except Exception:
                pass

            # Fallback to try other candidates if <speech> completely fails
            candidates = ["<|im_start|>", "<|im_end|>", "audio"]
            for text in candidates:
                try:
                    pid = tok.convert_tokens_to_ids(text)
                    if isinstance(pid, int) and pid != unk_id and pid > 0:
                        return int(pid)
                except Exception:
                    continue

        # Final hard fallback - use a safe token ID that should exist
        return 151646  # Should be the next available ID after <|im_end|>

    def remove_duplicate_token_batch(self, input_ids):
        target = 151646

        # —— 1. 压缩连续 target ——  
        same = (input_ids[1:] == target) & (input_ids[:-1] == target)
        keep = torch.ones_like(input_ids, dtype=torch.bool)
        keep[1:] = ~same
        compact = input_ids[keep]

        # —— 2. 分隔符（放到跟 compact 一样的 device） ——  
        sep = torch.tensor([
            151644,872,198,151646,14880,46670,61443,
            111268,17714,87335,151645,198,151644,77091,198
        ], device=compact.device)

        segments = []
        cur = []

        i = 0
        L = len(sep)

        # —— 3. 遍历并切分 ——  
        while i < len(compact):
            if i + L <= len(compact) and torch.equal(compact[i:i+L], sep):
                if cur:
                    segments.append(torch.tensor(cur, device=compact.device))
                    cur = []
                segments.append(sep.clone())  # sep 本身已在正确 device 上
                i += L
            else:
                cur.append(int(compact[i]))
                i += 1

        if cur:
            segments.append(torch.tensor(cur, device=compact.device))

        return segments

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        feature_lengths: Optional[torch.Tensor] = None,
        **kwargs: object,
    ) -> Union[torch.Tensor, IntermediateTensors]:


        if intermediate_tensors is not None:
            inputs_embeds = None
        hidden_states = self.language_model.model(
            input_ids,
            positions,
            intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )
        return hidden_states

    @classmethod
    def get_generation_prompt(
        cls,
        audio: np.ndarray,
        stt_config: SpeechToTextConfig,
        model_config: ModelConfig,
        language: Optional[str],
        task_type: Literal["transcribe", "translate"],
        request_prompt: str,
        to_language: Optional[str],
        decode: bool = False,
    ):
        """
        Generate the prompt for FireRedASR model using the input template from the original FireRedASR implementation.
        This function integrates the template logic from FireRedASR/fireredasr/tokenizer/llm_tokenizer.py
        """
        
        # Clean the request prompt if provided, otherwise use default
        if request_prompt and request_prompt.strip():
            clean_text = cls._clean_text(request_prompt.strip())
            content = f"{DEFAULT_SPEECH_TOKEN}{clean_text}"
        else:
            content = f"{DEFAULT_SPEECH_TOKEN}请转写音频为文字"

        # Create message structure following FireRedASR format
        messages = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": "" if decode else ""},
        ]

        base_prompt = cls._manual_template_construction(messages, decode)

        prompt = {
            "prompt": base_prompt,
            "multi_modal_data": {
                "audio": (audio, stt_config.sample_rate),
            }
        }

        return prompt

    @classmethod
    def _clean_text(cls, origin_text: str) -> str:
        """
        Clean text following the same logic as FireRedASR's LlmTokenizerWrapper.clean_text
        Remove punctuation, merge spaces, and handle Chinese/English spacing
        """
        # Remove punctuation
        text = re.sub("[，。？！,\\.!?《》（）\\·""、\\/]", "", origin_text)
        # Merge spaces
        text = re.sub("\\s+", " ", text)

        # Remove space between Chinese and keep space between English
        pattern = re.compile(r'([\u3400-\u4dbf\u4e00-\u9fff])')  # Chinese
        parts = pattern.split(text.strip())
        parts = [p for p in parts if len(p.strip()) > 0]
        text = "".join(parts)
        text = text.strip()

        text = text.lower()
        return text

    @classmethod
    def _manual_template_construction(cls, messages, decode: bool = False) -> str:
        """
        Manually construct the template when tokenizer.apply_chat_template is not available
        """
        result_parts = []
        for i, message in enumerate(messages):
            role = message["role"]
            content = message["content"]

            # Start each message with <|im_start|>role\ncontent
            result_parts.append(f"<|im_start|>{role}\n{content}")

            # Add ending based on position and decode flag
            if i == len(messages) - 1:  # Last message
                if not decode:
                    result_parts.append("")
                # For decode mode, don't add anything for the last message
            else:
                result_parts.append("<|im_end|>\n")

        return "".join(result_parts)


    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states)

    @classmethod
    def get_speech_to_text_config(
        cls, model_config: ModelConfig, task_type: str
    ) -> SpeechToTextConfig:
        """Get the speech to text config for FireRedASR model."""
        return SpeechToTextConfig(
            sample_rate=16000,  # FireRedASR expects 16kHz audio input
            max_audio_clip_s=30,  # Maximum audio clip duration in seconds
            overlap_chunk_second=1,  # Overlap between audio chunks
            min_energy_split_window_size=1600,  # 100ms at 16kHz for smart chunking
        )



    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        HF_LLM_CORE_PREFIX = 'model.'
        TARGET_LLM_CORE_PREFIX = 'language_model.' + HF_LLM_CORE_PREFIX
        
        HF_LM_HEAD_PREFIX = 'lm_head.'
        TARGET_LM_HEAD_PREFIX = 'language_model.' + HF_LM_HEAD_PREFIX
        
        # 1. 准备新的权重迭代器 (处理 Qwen2 前缀映射)
        modified_weights = []
        for name, weight in weights:
            # A. 替换 LLM 核心权重 (Qwen2Model)
            if name.startswith(HF_LLM_CORE_PREFIX):
                new_name = TARGET_LLM_CORE_PREFIX + name[len(HF_LLM_CORE_PREFIX):]
                modified_weights.append((new_name, weight))
                
            # B. 替换 LM Head 权重
            elif name.startswith(HF_LM_HEAD_PREFIX):
                new_name = TARGET_LM_HEAD_PREFIX + name[len(HF_LM_HEAD_PREFIX):]
                modified_weights.append((new_name, weight))
            
            # C. 处理 LLM 模块的其他顶层权重 (如 embed_tokens, norm)
            elif any(name.startswith(p) for p in ['embed_tokens.', 'norm.']):
                new_name = 'language_model.' + name
                modified_weights.append((new_name, weight))
                
            else:
                # 保留其他权重 (如 vLLM 配置项等)
                modified_weights.append((name, weight))
                
        # 2. 使用 AutoWeightsLoader 加载修改后的 LLM 权重
        loader = AutoWeightsLoader(self)
        # loaded_keys 包含所有通过 modified_weights 迭代器成功加载的 Qwen2 键
        loaded_keys = loader.load_weights(modified_weights)
        
        # 收集 speech_encoder 的所有权重名称
        if hasattr(self, 'speech_encoder'):
            for name, _ in self.speech_encoder.named_parameters():
                full_name = f'speech_encoder.{name}'
                loaded_keys.add(full_name)

        # 收集 multi_modal_projector 的所有权重名称
        if hasattr(self, 'multi_modal_projector'):
            for name, _ in self.multi_modal_projector.named_parameters():
                full_name = f'multi_modal_projector.{name}'
                loaded_keys.add(full_name)
        
        # 4. 返回包含 LLM 和 ASR 所有参数的集合
        return loaded_keys