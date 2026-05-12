# DiaMoE-TTS Minimal IPA Inference

This is a trimmed DiaMoE-TTS inference project. It keeps only the core path:

```text
IPA text -> ipa_tts_minimal model -> WAV audio
```

The original Gradio WebUI, dialect frontend, training scripts, documents, and demo pages have been removed.

## What Is Kept

- `core_infer.py`: minimal command-line inference entrypoint
- `diamoe_tts/src/f5_tts/`: required model and inference code
- `diamoe_tts/data/vocab.txt`: IPA/token vocabulary
- `prompts/zhengzhou_male_prompt.wav`: default reference voice
- `prompts/zhengzhou_male_prompt.txt`: default reference IPA text
- `checkpoints/`: local model checkpoint and vocoder files

## Required Files

The following files are required at runtime:

```text
checkpoints/10ep_mlpEXP_9.pt
checkpoints/vocos-mel-24khz/config.yaml
checkpoints/vocos-mel-24khz/pytorch_model.bin
diamoe_tts/data/vocab.txt
prompts/zhengzhou_male_prompt.wav
prompts/zhengzhou_male_prompt.txt
```

Large model files are ignored by Git by default. Keep them locally or download/copy them into the paths above before running inference.

## Install

From the project root:

```bash
cd /mnt/d/xinghua/DiaMoE-TTS
python3 -m pip install -r diamoe_tts/requirements.txt
```

If you use Windows instead of Ubuntu/WSL:

```powershell
cd D:\xinghua\DiaMoE-TTS
python -m pip install -r diamoe_tts\requirements.txt
```

## Run

Ubuntu/WSL:

```bash
cd /mnt/d/xinghua/DiaMoE-TTS
python3 core_infer.py --ipa "t a | n i" --output outputs/test.wav
```

Windows:

```powershell
cd D:\xinghua\DiaMoE-TTS
python core_infer.py --ipa "t a | n i" --output outputs\test.wav
```

The output file will be written to:

```text
outputs/test.wav
```

## IPA Input Format

Separate every IPA token with spaces:

```bash
python3 core_infer.py --ipa "t a | n i" --output outputs/test.wav
```

You may also provide bracketed tokens:

```bash
python3 core_infer.py --ipa "[t] [a] | [n] [i]" --output outputs/test.wav
```

Do not write tokens without spaces:

```text
[ni3][hao3]
```

That is treated as one token and is also pinyin, not IPA. This minimal version does not include text-to-pinyin, pinyin-to-IPA, or dialect selection.

## IPA From File

```bash
python3 core_infer.py --ipa-file my_ipa.txt --output outputs/result.wav
```

`my_ipa.txt` should contain whitespace-separated IPA tokens.

## Optional Arguments

```bash
python3 core_infer.py \
  --ipa "t a | n i" \
  --output outputs/test.wav \
  --ref-audio prompts/zhengzhou_male_prompt.wav \
  --ref-ipa-file prompts/zhengzhou_male_prompt.txt \
  --speed 1.0 \
  --steps 32
```

## Notes

- Very short IPA input may produce very short or quiet audio. Use a longer IPA sequence for real testing.
- The default reference voice is Zhengzhou male.
- This project is for inference only. Training and WebUI code were intentionally removed.
