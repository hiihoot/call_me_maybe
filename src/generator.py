"""Generator: natural language → structured function calls."""
import json
from typing import Any, Dict, List

import numpy as np
try:
    from llm_sdk import Small_LLM_Model  # type: ignore [attr-defined]
except KeyboardInterrupt:
    exit(0)


class Generator:
    """One LLM call picks the function;
    iterative JSON completion extracts parameters.
    """

    def __init__(self, definitions: Any) -> None:
        self.llm = Small_LLM_Model()
        self.funcs: Dict[str, Dict[str, Any]] = {}
        for d in (definitions if isinstance(definitions, list)
                  else [definitions]):
            item = (d.model_dump() if hasattr(d, "model_dump")
                    else d.dict() if hasattr(d, "dict") else d)
            if item.get("name"):
                self.funcs[item["name"]] = item
        self.names = list(self.funcs.keys())
        self.vocab = self.load_vocab()

    def load_vocab(self) -> Dict[int, str]:
        with open(self.llm.get_path_to_vocab_file(), "r") as f:
            vocab = json.load(f)
        first = next(iter(vocab.keys()))
        it = ({int(v): k for k, v in vocab.items()} if isinstance(first, str)
              else {int(k): v for k, v in vocab.items()})
        return ({tid: tok.replace("Ġ", " ").replace("Ċ", "\n")
                 for tid, tok in it.items()})

    def encode(self, text: str) -> List[int]:
        raw = self.llm.encode(text)
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        return [int(x) for x in (raw[0]
                                 if raw and isinstance(raw[0], list) else raw)]

    def valid_exact(self, prefix: str, allowed: List[str]) -> List[int]:
        return [tid for tid, tok in self.vocab.items()
                if any(a.startswith((prefix + tok).lstrip()) or
                       (prefix + tok).lstrip().startswith(a) for a in allowed)]

    def constrained_exact(
            self, ctx: List[int],
            allowed: List[str],
            max_len: int = 15) -> str:
        """Forces output to exactly match
        one string from the 'allowed' list.
        """
        cur, toks = "", list(ctx)
        for _ in range(max_len):
            # dtype=np.float32
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
            # dtype=np.float32
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
            # Stop generation exactly at the JSON boundary
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
            # dtype=np.float32
            logits = np.array(self.llm.get_logits_from_input_ids(toks))
            # Strings can be anything, so we use standard
            # argmax without a restrictive mask
            tid = int(np.argmax(logits))
            tok_str = self.vocab.get(tid, "")
            if '"' in tok_str:
                idx = tok_str.find('"')
                # If the quote is escaped (\"), keep going. Otherwise, break.
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

    def props(self, func_def: Dict[str, Any]) -> Any:
        p = func_def.get("parameters", {})
        if "properties" in p:
            return p["properties"]
        if isinstance(p, dict) and any(isinstance(v, dict) and
                                       "type" in v for v in p.values()):
            return p
        return {}

    def run_prompt(self, user_prompt: str) -> Dict[str, Any]:
        if not self.funcs:
            return {"prompt": user_prompt, "name": "", "parameters": {}}

        # 1. Function Routing
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
        # 2. Iterative JSON Construction
        # We build a fake JSON block and append it to the prompt.
        # This forces the LLM to complete
        # the data structure instead of answering a question.
        json_prefix = (f'User: {user_prompt}\nCall:'
                       f'{func_name}\nArguments:\n{{\n')
        for pname, pschema in props.items():
            ptype = pschema.get("type", "string")
            enums = pschema.get("enum", [])
            # Start asking for the parameter
            param_prompt = json_prefix + f'  "{pname}": '

            if enums:
                quoted_enums = [f'"{e}"' for e in enums]
                ctx = self.encode(param_prompt)
                val = self.constrained_exact(ctx, quoted_enums)
                val = val.strip('"')
                params[pname] = val
                json_prefix += f'  "{pname}": "{val}",\n'

            elif ptype == "boolean":
                ctx = self.encode(param_prompt)
                val = self.constrained_exact(ctx, ["true", "false"])
                params[pname] = (val == "true")
                json_prefix += f'  "{pname}": {val},\n'

            elif ptype in ("integer", "number"):
                ctx = self.encode(param_prompt)
                val_str = self.constrained_number(ctx)
                clean = val_str.strip()

                if ptype == "integer":
                    params[pname] = (int(clean) if clean.lstrip('-').isdigit()
                                     else 0)
                else:
                    try:
                        params[pname] = float(clean)
                    except ValueError:
                        params[pname] = 0.0

                # Feed the successful extraction back
                # into the prompt for the next parameter
                json_prefix += f'  "{pname}": {params[pname]},\n'

            else:  # string
                # We physically add the opening quote
                # to the prompt to force a string output
                ctx = self.encode(param_prompt + '"')
                val_str = self.constrained_string(ctx)
                params[pname] = val_str

                json_prefix += f'  "{pname}": "{val_str}",\n'

        return {"prompt": user_prompt, "name": func_name, "parameters": params}
