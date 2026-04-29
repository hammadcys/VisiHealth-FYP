import json
import traceback

status = {"ok": False}

try:
    import scripts.refresh_model_artifacts as refresh
    refresh.main()
    status["ok"] = True
except Exception as exc:
    status["error"] = repr(exc)
    status["traceback"] = traceback.format_exc()

with open("refresh_status.json", "w", encoding="utf-8") as f:
    json.dump(status, f, indent=2)

print("Wrote refresh_status.json")
