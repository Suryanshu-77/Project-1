"""Export MiniLM to ONNX for the learned router.

Usage: uv run --group train python -m training.export_onnx [--out models/router]
Writes model.onnx (token embeddings, dynamic batch and sequence axes), and tokenizer.json if absent.
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

ENCODER = "sentence-transformers/all-MiniLM-L6-v2"
INPUTS = ("input_ids", "attention_mask", "token_type_ids")


class _TokenEmbeddings(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):  # type: ignore[no-untyped-def]
        out = self.model(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        return out.last_hidden_state


def export(out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    model = _TokenEmbeddings(AutoModel.from_pretrained(ENCODER)).eval()
    tokenizer = AutoTokenizer.from_pretrained(ENCODER)
    sample = tokenizer(["a short example", "a slightly longer example sentence"],
                       padding=True, return_tensors="pt")  # fmt: skip
    dynamic = {name: {0: "batch", 1: "sequence"} for name in (*INPUTS, "last_hidden_state")}
    with torch.no_grad():
        torch.onnx.export(
            model,
            tuple(sample[name] for name in INPUTS),
            str(out / "model.onnx"),
            input_names=list(INPUTS),
            output_names=["last_hidden_state"],
            dynamic_axes=dynamic,
            opset_version=17,
            dynamo=False,
        )
    if not (out / "tokenizer.json").exists():  # the committed copy is the one the gateway uses
        tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))
    print(f"wrote {out / 'model.onnx'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("models/router"))
    export(parser.parse_args().out)
