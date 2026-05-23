from __future__ import annotations

import logging

import torch

import config

log = logging.getLogger(__name__)


class Generator:
    def __init__(self, model_name: str = config.GENERATION_MODEL) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.model_name = model_name
        log.info("Loading tokenizer: %s", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        dtype = torch.bfloat16 if config.TORCH_DTYPE == "bfloat16" else torch.float16

        if config.LOAD_IN_4BIT:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                device_map=config.DEVICE_MAP,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=config.DEVICE_MAP,
            )

        self.model.eval()
        log.info("Model loaded: %s", model_name)

    def generate(self, messages: list[dict]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages)
            prompt += "\nASSISTANT:"

        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"]

        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)

        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        do_sample = config.TEMPERATURE > 0
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=config.MAX_NEW_TOKENS,
                do_sample=do_sample,
                temperature=config.TEMPERATURE if do_sample else None,
                top_p=config.TOP_P if do_sample else None,
                pad_token_id=pad_token_id,
            )

        new_tokens = output_ids[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def unload(self) -> None:
        try:
            del self.model
        except Exception:
            pass
        try:
            del self.tokenizer
        except Exception:
            pass
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
