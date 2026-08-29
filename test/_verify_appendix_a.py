from __future__ import annotations
import re, sys
from pathlib import Path
ROOT = Path(r"D:/USERFILES/GitHub/ComfyUI-HSWQ-Loader-and-Tools")
BT = chr(96)
HEAD_RE = re.compile(r"^### " + BT + r"([^" + BT + r"]+)" + BT + r"\s*$", re.M)
FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)

def check(guide, marker):
    text = guide.read_text(encoding="utf-8")
    idx = text.find(marker)
    if idx < 0:
        print("FAIL no marker", marker, "in", guide.name)
        return set(), 1
    text = text[idx:]
    headings = list(HEAD_RE.finditer(text))
    fails = 0
    paths = []
    for i, h in enumerate(headings):
        path = h.group(1)
        paths.append(path)
        start = h.end()
        end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        body = text[start:end]
        m = FENCE_RE.search(body)
        if not m:
            print("  MISS fence:", path); fails += 1; continue
        embed = m.group(1).replace("\r\n", "\n")
        fp = ROOT / path
        if not fp.exists():
            print("  MISS disk:", path); fails += 1; continue
        disk = fp.read_text(encoding="utf-8").replace("\r\n", "\n")
        if disk == embed or disk.rstrip("\n") == embed.rstrip("\n"):
            print("  OK:", path, len(disk))
        else:
            de, ee = disk.rstrip("\n"), embed.rstrip("\n")
            pos = next((j for j,(a,b) in enumerate(zip(de,ee)) if a!=b), None)
            print("  MISMATCH:", path, "disk", len(de), "embed", len(ee), "@", pos)
            fails += 1
    return set(paths), fails

en_paths, en_f = check(ROOT/"md"/"HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md", "## Appendix A")
print("--- ZH ---")
zh_paths, zh_f = check(ROOT/"zhmd"/"HSWQ_FROM_a9d372_PROBLEM_COUNTERMEASURES_GUIDE.md", "## 附录 A")
print("EN only:", sorted(en_paths - zh_paths))
print("ZH only:", sorted(zh_paths - en_paths))
print("FAILS", en_f + zh_f)
