"""Generator: natural language → structured function calls."""
import json
from typing import Any, Dict, List, Optional

import numpy as np
from llm_sdk import Small_LLM_Model # ignore


class Generator:
    """One LLM call picks the function name; deterministic extraction fills params."""

    def __init__(self, definitions: Any) -> None:
        self.llm = Small_LLM_Model()
        self.funcs: Dict[str, Dict[str, Any]] = {}
        for d in (definitions if isinstance(definitions, list) else [definitions]):
            item = d.model_dump() if hasattr(d, "model_dump") else d.dict() if hasattr(d, "dict") else d
            if item.get("name"):
                self.funcs[item["name"]] = item
        self.names = list(self.funcs.keys())
        self.vocab = self._load_vocab()

    def _load_vocab(self) -> Dict[int, str]:
        with open(self.llm.get_path_to_vocab_file(), "r", encoding="utf-8") as f:
            vocab = json.load(f)
        first = next(iter(vocab.keys()))
        it = {int(v): k for k, v in vocab.items()} if isinstance(first, str) else {int(k): v for k, v in vocab.items()}
        return {tid: tok.replace("Ġ", " ").replace("Ċ", "\n") for tid, tok in it.items()}

    def _encode(self, text: str) -> List[int]:
        raw = self.llm.encode(text)
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        return [int(x) for x in (raw[0] if raw and isinstance(raw[0], list) else raw)]

    def _valid(self, prefix: str, allowed: List[str]) -> List[int]:
        return [tid for tid, tok in self.vocab.items()
                if any(a.startswith((prefix + tok).lstrip()) or (prefix + tok).lstrip().startswith(a) for a in allowed)]

    def _constrained(self, ctx: List[int], allowed: List[str], max_len: int = 20) -> str:
        cur, toks = "", list(ctx)
        for _ in range(max_len):
            logits = np.array(self.llm.get_logits_from_input_ids(toks), dtype=np.float32)
            valid = self._valid(cur, allowed)
            if not valid:
                break
            mask = np.full_like(logits, -np.inf)
            for tid in valid:
                mask[tid] = logits[tid]
            tid = int(np.argmax(mask))
            cur += self.vocab.get(tid, "")
            toks.append(tid)
            if cur.lstrip() in allowed:
                return cur.lstrip()
        return allowed[0] if allowed else ""

    def _pick_function(self, prompt: str) -> str:
        text = f'User Request: "{prompt}"\nWhich function should be called?\n'
        for name, spec in self.funcs.items():
            text += f"- {name}: {spec.get('description', '')}\n"
        return self._constrained(self._encode(text + "Function Name: "), self.names, 15)

    def _nums(self, text: str) -> List[float]:
        out, i, n = [], 0, len(text)
        while i < n:
            if text[i] == "-" or text[i].isdigit():
                j = i + (1 if text[i] == "-" else 0)
                if j < n and text[j].isdigit():
                    s = ("-" if text[i] == "-" else "")
                    i += len(s)
                    while i < n and text[i].isdigit():
                        s += text[i]; i += 1
                    if i < n and text[i] == ".":
                        s += "."; i += 1
                        while i < n and text[i].isdigit():
                            s += text[i]; i += 1
                    try:
                        out.append(float(s))
                    except ValueError:
                        pass
                    continue
            i += 1
        return out

    def _quoted(self, text: str) -> List[str]:
        out, i, n = [], 0, len(text)
        while i < n:
            if text[i] in ('"', "'"):
                q, start = text[i], i + 1
                i += 1
                while i < n and not (text[i] == q and text[i - 1] != "\\"):
                    i += 1
                out.append(text[start:i])
                i += 1
                continue
            i += 1
        return out

    def _path(self, text: str) -> Optional[str]:
        for i, ch in enumerate(text):
            if ch in ("/", "~"):
                j = i + 1
                while j < len(text) and text[j] not in " \t\n\"'":
                    j += 1
                return text[i:j]
            if i + 2 < len(text) and text[i + 1] == ":" and text[i + 2] == "\\" and ch.isalpha():
                j = i + 3
                while j < len(text) and text[j] not in " \t\n\"'":
                    j += 1
                return text[i:j]
        return None

    def _by_name(self, prompt: str, pname: str) -> Optional[str]:
        low, nl = prompt.lower(), pname.lower()
        ci = low.find(nl + ":")
        if ci != -1:
            return prompt[ci + len(pname) + 1:].strip()
        idx = low.find(nl)
        if idx != -1:
            rest = prompt[idx + len(pname):]
            for q in ("'", '"'):
                s = rest.find(q)
                if s != -1:
                    e = s + 1
                    while e < len(rest):
                        if rest[e] == "\\":
                            e += 2
                        elif rest[e] == q:
                            return rest[s + 1:e]
                        else:
                            e += 1
            words = prompt[:idx].rstrip().split()
            if words:
                w = words[-1]
                if w.lower() in {"the", "a", "an"} and len(words) > 1:
                    w = words[-2]
                return w
        return None

    def _props(self, func_def: Dict[str, Any]) -> Dict[str, Any]:
        p = func_def.get("parameters", {})
        if "properties" in p:
            return p["properties"]
        if isinstance(p, dict) and any(isinstance(v, dict) and "type" in v for v in p.values()):
            return p
        return {}

    def run_prompt(self, user_prompt: str) -> Dict[str, Any]:
        if not self.funcs:
            return {"prompt": user_prompt, "name": "", "parameters": {}}

        try:
            func_name = self._pick_function(user_prompt)
        except Exception:
            func_name = self.names[0] if self.names else ""

        props = self._props(self.funcs.get(func_name, {}))
        params: Dict[str, Any] = {}
        numbers = self._nums(user_prompt)
        quoted = self._quoted(user_prompt)
        lower = user_prompt.lower()
        ni = si = 0

        for pname, pschema in props.items():
            ptype = pschema.get("type", "string")
            enums = pschema.get("enum", [])

            if enums:
                pick = next((v for v in enums if v.lower() in lower), None)
                params[pname] = pick if pick else enums[0]
            elif ptype == "boolean":
                params[pname] = "true" in lower or "yes" in lower
            elif ptype in ("integer", "number"):
                if ni < len(numbers):
                    v = numbers[ni]
                    ni += 1
                    params[pname] = int(v) if ptype == "integer" else v
                else:
                    params[pname] = 0 if ptype == "integer" else 0.0
            elif ptype == "string":
                val = self._by_name(user_prompt, pname)
                if val is not None:
                    if val in quoted:
                        quoted.remove(val)
                    params[pname] = val
                    continue

                if any(k in pname.lower() for k in ("path", "file")):
                    pval = self._path(user_prompt)
                    if pval is not None:
                        params[pname] = pval
                        continue

                if si < len(quoted):
                    params[pname] = quoted[si]
                    si += 1
                    continue

                for w in reversed(user_prompt.split()):
                    c = w.strip(".,!?;:'\"()[]{}")
                    if c and not any(ch.isdigit() for ch in c):
                        params[pname] = c
                        break
                else:
                    params[pname] = ""
            else:
                params[pname] = None

        return {"prompt": user_prompt, "name": func_name, "parameters": params}