#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import soundfile as sf
from hydra.utils import get_class
from omegaconf import OmegaConf

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR / "diamoe_tts" / "src"))


DEFAULT_CKPT = BASE_DIR / "checkpoints" / "10ep_mlpEXP_9.pt"
DEFAULT_VOCAB = BASE_DIR / "diamoe_tts" / "data" / "vocab.txt"
DEFAULT_CONFIG = BASE_DIR / "diamoe_tts" / "src" / "f5_tts" / "configs" / "gradio.yaml"
DEFAULT_VOCODER = BASE_DIR / "checkpoints" / "vocos-mel-24khz"
DEFAULT_REF_AUDIO = BASE_DIR / "prompts" / "zhengzhou_male_prompt.wav"
DEFAULT_REF_TEXT = BASE_DIR / "prompts" / "zhengzhou_male_prompt.txt"
DEFAULT_OUTPUT = BASE_DIR / "outputs" / "ipa_output.wav"
MEL_SPEC_TYPE = "vocos"
TARGET_RMS = 0.1
CROSS_FADE_DURATION = 0.15
NFE_STEP = 32
CFG_STRENGTH = 2.0
SWAY_SAMPLING_COEF = -1.0

PUNCTUATION = {
    ",",
    ".",
    "!",
    "?",
    ";",
    ":",
    "|",
    "，",
    "。",
    "！",
    "？",
    "；",
    "：",
    "、",
}


def _resolve(path: str | os.PathLike | None, default: Path | None = None) -> Path | None:
    if not path:
        return default
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = BASE_DIR / value
    return value


def load_prompt_ipa(prompt_txt: str | os.PathLike = DEFAULT_REF_TEXT) -> str:
    """Read the IPA column from a bundled prompt text file."""
    line = Path(prompt_txt).read_text(encoding="utf-8").strip()
    parts = line.split("\t")
    if len(parts) >= 3:
        return parts[2].strip()
    return line


def normalize_ipa(ipa_text: str) -> str:
    """
    Convert a plain whitespace-separated IPA sequence into DiaMoE-TTS token form.

    The model keeps IPA units as bracketed tokens, for example:
        t͜ɕ ˈiᴹᴸ | u -> [t͜ɕ] [ˈiᴹᴸ] | [u]

    If the input already contains bracketed tokens, they are left unchanged.
    """
    if not ipa_text or not ipa_text.strip():
        raise ValueError("IPA input is empty.")

    normalized = []
    for token in ipa_text.strip().split():
        if token in PUNCTUATION:
            normalized.append(token)
        elif token.startswith("[") and token.endswith("]"):
            normalized.append(token)
        else:
            normalized.append(f"[{token}]")
    return " ".join(normalized)


class DiaMoETTS:
    """Minimal IPA-in, WAV-out DiaMoE-TTS inference wrapper."""

    def __init__(
        self,
        ckpt_file: str | os.PathLike = DEFAULT_CKPT,
        vocab_file: str | os.PathLike = DEFAULT_VOCAB,
        config_file: str | os.PathLike = DEFAULT_CONFIG,
        vocoder_path: str | os.PathLike = DEFAULT_VOCODER,
    ) -> None:
        self.ckpt_file = Path(ckpt_file)
        self.vocab_file = Path(vocab_file)
        self.config_file = Path(config_file)
        self.vocoder_path = Path(vocoder_path)
        self.model = None
        self.vocoder = None
        self.device = None

    def load(self) -> None:
        if self.model is not None and self.vocoder is not None:
            return

        from f5_tts.infer.utils_infer import device, load_model, load_vocoder

        model_cfg = OmegaConf.load(self.config_file)
        model_cls = get_class(f"f5_tts.model.{model_cfg.model.backbone}")
        model_arc = model_cfg.model.arch
        self.device = device

        self.vocoder = load_vocoder(
            vocoder_name=MEL_SPEC_TYPE,
            is_local=self.vocoder_path.exists(),
            local_path=str(self.vocoder_path),
            device=self.device,
        )
        self.model = load_model(
            model_cls,
            model_arc,
            str(self.ckpt_file),
            mel_spec_type=MEL_SPEC_TYPE,
            vocab_file=str(self.vocab_file),
            device=self.device,
            use_moe=True,
            num_exps=9,
            moe_topK=1,
            expert_type="mlp",
        )

    def synthesize(
        self,
        ipa_text: str,
        output_path: str | os.PathLike = DEFAULT_OUTPUT,
        ref_audio: str | os.PathLike | None = DEFAULT_REF_AUDIO,
        ref_ipa: str | None = None,
        ref_ipa_file: str | os.PathLike | None = DEFAULT_REF_TEXT,
        speed: float = 1.0,
        steps: int = NFE_STEP,
    ) -> Path:
        self.load()

        from f5_tts.infer.utils_infer import infer_process, preprocess_ref_audio_text

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        gen_text = normalize_ipa(ipa_text)
        ref_audio = ref_audio or DEFAULT_REF_AUDIO
        if ref_ipa is None:
            ref_ipa = load_prompt_ipa(ref_ipa_file or DEFAULT_REF_TEXT)
        ref_text = normalize_ipa(ref_ipa)

        ref_audio_path, ref_text = preprocess_ref_audio_text(str(ref_audio), ref_text)
        wav, sample_rate, _ = infer_process(
            ref_audio_path,
            ref_text,
            gen_text,
            self.model,
            self.vocoder,
            mel_spec_type=MEL_SPEC_TYPE,
            show_info=lambda *_args, **_kwargs: None,
            progress=None,
            target_rms=TARGET_RMS,
            cross_fade_duration=CROSS_FADE_DURATION,
            nfe_step=steps,
            cfg_strength=CFG_STRENGTH,
            sway_sampling_coef=SWAY_SAMPLING_COEF,
            speed=speed,
            fix_duration=None,
            device=self.device,
        )
        sf.write(output, wav, sample_rate)
        return output


def synthesize_ipa(
    ipa_text: str,
    output_path: str | os.PathLike = DEFAULT_OUTPUT,
    ref_audio: str | os.PathLike = DEFAULT_REF_AUDIO,
    ref_ipa: str | None = None,
    ref_ipa_file: str | os.PathLike | None = DEFAULT_REF_TEXT,
    speed: float = 1.0,
    steps: int = NFE_STEP,
) -> Path:
    tts = DiaMoETTS()
    return tts.synthesize(
        ipa_text=ipa_text,
        output_path=output_path,
        ref_audio=ref_audio,
        ref_ipa=ref_ipa,
        ref_ipa_file=ref_ipa_file,
        speed=speed,
        steps=steps,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="DiaMoE-TTS minimal IPA inference.")
    parser.add_argument("--ipa", help="Whitespace-separated IPA input.")
    parser.add_argument("--ipa-file", help="Text file containing IPA input.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output wav path.")
    parser.add_argument("--ref-audio", default=str(DEFAULT_REF_AUDIO), help="Reference audio path.")
    parser.add_argument("--ref-ipa", help="Reference IPA text. Defaults to bundled prompt IPA.")
    parser.add_argument("--ref-ipa-file", default=str(DEFAULT_REF_TEXT), help="Reference IPA text file.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed.")
    parser.add_argument("--steps", type=int, default=NFE_STEP, help="Diffusion denoising steps.")
    args = parser.parse_args()

    if args.ipa_file:
        ipa_text = Path(args.ipa_file).read_text(encoding="utf-8").strip()
    elif args.ipa:
        ipa_text = args.ipa
    else:
        raise SystemExit("Please pass --ipa or --ipa-file.")

    out = synthesize_ipa(
        ipa_text=ipa_text,
        output_path=args.output,
        ref_audio=args.ref_audio,
        ref_ipa=args.ref_ipa,
        ref_ipa_file=args.ref_ipa_file,
        speed=args.speed,
        steps=args.steps,
    )
    print(f"Generated: {out}")


if __name__ == "__main__":
    main()
