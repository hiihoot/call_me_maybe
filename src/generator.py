"""Generator: natural language → structured function calls."""
import json
from typing import Any, Dict, List

import numpy as np
try:
    from llm_sdk import Small_LLM_Model  # type: ignore [attr-defined]
except KeyboardInterrupt:
    exit(0)


class Generator:
    """
    One LLM call picks the function;
    iterative JSON completion extracts parameters.
    """
    def __init__(self, definitions: Any) -> None:
        self.llm = Small_LLM_Model()
        self.funcs: Dict[str, Dict[str, Any]] = {}
        for d in (definitions):
            item = d.model_dump()
            if item.get("name"):
                self.funcs[item["name"]] = item
        self.names = list(self.funcs.keys())
        self.vocab = self.load_vocab()

    def load_vocab(self) -> Dict[int, str]:
        with open(self.llm.get_path_to_vocab_file(), "r") as f:
            vocab = json.load(f)
        return ({tid: tok.replace("Ġ", " ").replace("Ċ", "\n")
                 for tok, tid in vocab.items()})

    def encode(self, text: str) -> List[int]:
        raw = self.llm.encode(text)
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if raw and isinstance(raw[0], list):
            raw = raw[0]
        return [int(x) for x in raw]

    def valid_exact(self, prefix: str, allowed: List[str]) -> List[int]:
        valid_ids = []

        for tid, tok in self.vocab.items():
            candidate_text = (prefix + tok).lstrip()
            for target in allowed:
                if (target.startswith(candidate_text)
                        or candidate_text.startswith(target)):
                    valid_ids.append(tid)
                    break

        return valid_ids

    def constrained_exact(
            self, ctx: List[int],
            allowed: List[str],
            max_len: int = 15) -> str:
        """Forces output to exactly match
        one string from the 'allowed' list.
        """
        cur, toks = "", list(ctx)
        for _ in range(max_len):
            logits = np.array(self.llm.get_logits_from_input_ids(toks))
            valid = self.valid_exact(cur, allowed)
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

    def constrained_number(self, ctx: List[int], max_len: int = 15) -> str:
        """Generates numbers by masking out all alphabetical characters."""
        cur, toks = "", list(ctx)
        for _ in range(max_len):
            logits = np.array(self.llm.get_logits_from_input_ids(toks))
            valid = []
            for tid, tok in self.vocab.items():
                clean = tok.strip()
                if not clean or all(c.isdigit() or
                                    c in "-.,}\n" for c in clean):
                    valid.append(tid)
            if not valid:
                break
            mask = np.full_like(logits, -np.inf)
            for tid in valid:
                mask[tid] = logits[tid]
            tid = int(np.argmax(mask))
            tok_str = self.vocab.get(tid, "")
            if any(c in tok_str for c in ",\n}"):
                idx = min([tok_str.find(c) for c in ",\n}" if c in tok_str])
                cur += tok_str[:idx]
                break
            cur += tok_str
            toks.append(tid)
        return cur

    def constrained_string(self, ctx: List[int], max_len: int = 50) -> str:
        """Generates an open string
        stopping only at an unescaped closing quote.
        """
        cur, toks = "", list(ctx)
        for _ in range(max_len):
            logits = np.array(self.llm.get_logits_from_input_ids(toks))
            tid = int(np.argmax(logits))
            tok_str = self.vocab.get(tid, "")
            if '"' in tok_str:
                idx = tok_str.find('"')
                if idx > 0 and tok_str[idx - 1] == '\\':
                    cur += tok_str
                    toks.append(tid)
                else:
                    cur += tok_str[:idx]
                    break
            else:
                cur += tok_str
                toks.append(tid)
        return cur

    def props(self, func_def: Dict[str, Any]) -> Dict[str, Any]:
        parameters = func_def.get("parameters", {})
        if isinstance(parameters, dict):
            for value in parameters.values():
                if isinstance(value, dict) and "type" in value:
                    return parameters

        return {}

    def run_prompt(self, user_prompt: str) -> Dict[str, Any]:
        prompt_fn = f'User Request: "{user_prompt}"\nFunctions:\n'
        for name, spec in self.funcs.items():
            prompt_fn += f"- {name}: {spec.get('description', '')}\n"
        prompt_fn += 'Target Function: '
        try:
            func_name = self.constrained_exact(
                self.encode(prompt_fn),
                self.names, 15
            )
        except Exception:
            func_name = self.names[0] if self.names else ""

        props = self.props(self.funcs.get(func_name, {}))
        params: Dict[str, Any] = {}
        json_prefix = (f'User: {user_prompt}\nCall:'
                       f'{func_name}\nArguments:\n{{\n')

        for pname, pschema in props.items():
            ptype = pschema.get("type", "string")
            param_prompt = json_prefix + f'  "{pname}": '

            if ptype == "boolean":
                ctx = self.encode(param_prompt)
                val = self.constrained_exact(ctx, ["true", "false"])
                params[pname] = (val == "true")
                json_prefix += f'  "{pname}": {val},\n'

            elif ptype in ("integer", "number"):
                ctx = self.encode(param_prompt)
                val_str = self.constrained_number(ctx)
                clean = val_str.strip()

                if ptype == "integer":
                    params[pname] = (int(clean) if clean.lstrip().isdigit()
                                     else 0)
                else:
                    try:
                        params[pname] = float(clean)
                    except ValueError:
                        params[pname] = 0.0
                json_prefix += f'  "{pname}": {params[pname]},\n'

            else:
                ctx = self.encode(param_prompt + '"')
                val_str = self.constrained_string(ctx)
                params[pname] = val_str

                json_prefix += f'  "{pname}": "{val_str}",\n'

        return {"prompt": user_prompt, "name": func_name, "parameters": params}
