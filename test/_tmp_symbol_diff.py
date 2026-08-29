"""One-shot: list defs only in HEAD backup vs a9d3720 tree."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def defs(p: Path):
    t = ast.parse(p.read_text(encoding="utf-8"))
    return sorted(
        {
            n.name
            for n in t.body
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
    )


pairs = [
    ("conf", "nodes/nvfp4/nvfp4_conf.py", "_sdxl_restore_backup/nvfp4_conf.py"),
    ("fwd", "nodes/nvfp4/nvfp4_forward.py", "_sdxl_restore_backup/nvfp4_forward.py"),
    ("had", "nodes/nvfp4/nvfp4_hadamard.py", "_sdxl_restore_backup/nvfp4_hadamard.py"),
    ("cq", "nodes/nvfp4/comfy_quant_nvfp4.py", "_sdxl_restore_backup/comfy_quant_nvfp4.py"),
]
for name, a, b in pairs:
    da, db = set(defs(ROOT / a)), set(defs(ROOT / b))
    print(name, "ONLY_IN_HEAD_BACKUP", sorted(db - da))
    print(name, "ONLY_IN_A9", sorted(da - db))
