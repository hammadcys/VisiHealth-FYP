import json
import traceback

result = {"ok": False}

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("michiyasunaga/BioLinkBERT-base")
    result["ok"] = True
    result["tokenizer_class"] = tok.__class__.__name__
except Exception as exc:
    result["error"] = repr(exc)
    result["traceback"] = traceback.format_exc()

with open("hf_test_result.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2)

print("Wrote hf_test_result.json")
