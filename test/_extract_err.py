import json
import io

PATH = r"C:\Users\ussoe\.cursor\projects\d-USERFILES-GitHub-ComfyUI-HSWQ-Loader-and-Tools\agent-transcripts\547a99f9-b81b-4956-8e06-dd46470bd034\547a99f9-b81b-4956-8e06-dd46470bd034.jsonl"
KEY = "not subscriptable"

out = io.open(r"d:\USERFILES\GitHub\ComfyUI-HSWQ-Loader-and-Tools\test\_err_ctx.txt", "w", encoding="utf-8")
with io.open(PATH, "r", encoding="utf-8", errors="replace") as f:
    for n, line in enumerate(f, 1):
        if KEY not in line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if obj.get("role") != "user":
            continue
        parts = obj.get("message", {}).get("content", [])
        for p in parts:
            t = p.get("text", "")
            idx = 0
            while True:
                i = t.find(KEY, idx)
                if i < 0:
                    break
                out.write("=== line %d @%d ===\n" % (n, i))
                out.write(t[max(0, i - 3000): i + 500])
                out.write("\n\n")
                idx = i + 1
out.close()
print("done")
