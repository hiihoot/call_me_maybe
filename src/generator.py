"""Generator class for constrained function calling with a small LLM."""
import json
from typing import Any, Dict, List, Optional

import numpy as np
from llm_sdk import Small_LLM_Model


class Generator:
    """Convert natural language prompts into structured function calls.

    Uses exactly one LLM call (constrained decoding) to select the function
    name, then extracts parameters deterministically from the prompt text.
    """

    REGEX_INFERENCE = {
        "all numbers": r"\d+",
        "any number": r"\d+",
        "all digits": r"\d+",
        "all vowels": "[aeiouAEIOU]",
        "any vowel": "[aeiouAEIOU]",
    }

    WORD_TO_SYMBOL = {
        "asterisks": "*",
        "asterisk": "*",
        "star": "*",
        "stars": "*",
        "percent": "%",
        "percentage": "%",
        "dollar": "$",
        "dollars": "$",
        "hash": "#",
        "hashes": "#",
        "at": "@",
    }

    PARAM_HINTS = {
        "replace": ["with", "to", "by"],
        "source":  ["in", "from", "of"],
        "regex":   ["word", "pattern", "text"],
        "target":  ["to", "into"],
        "path":    ["at", "in"],
    }

    _ARTICLES = {"the", "a", "an"}

    def __init__(self, definitions: Any) -> None:
        """Initialize the generator with function definitions.

        Args:
            definitions: A single function dict/model, or a list of them.

        Raises:
            FileNotFoundError: If the vocabulary file is missing.
            ValueError: If the vocabulary file contains invalid JSON.
            RuntimeError: If the LLM SDK fails during initialization.
        """
        try:
            self.llm = Small_LLM_Model()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize LLM: {exc}") from exc

        self.func_map: Dict[str, Dict[str, Any]] = {}
        self._parse_definitions(definitions)
        self.func_names: List[str] = list(self.func_map.keys())
        self.vocab_map: Dict[int, str] = self._load_vocab_map()

    def _parse_definitions(self, defs: Any) -> None:
        """Normalise arbitrary definition objects into a flat name → dict map."""
        if not isinstance(defs, list):
            defs = [defs]
        for item in defs:
            d: Dict[str, Any] = {}
            if hasattr(item, "model_dump"):
                d = item.model_dump()
            elif hasattr(item, "dict"):
                d = item.dict()
            elif isinstance(item, dict):
                d = item
            name = d.get("name")
            if name:
                self.func_map[name] = d

    def _load_vocab_map(self) -> Dict[int, str]:
        """Load the LLM vocabulary and normalise special tokens.

        Returns:
            Mapping from token id to decoded token string.

        Raises:
            FileNotFoundError: If the vocab file is missing.
            ValueError: If the vocab file contains invalid JSON.
        """
        vocab_path = self.llm.get_path_to_vocab_file()
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Vocabulary file not found: {vocab_path}"
            ) from None
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in vocabulary file {vocab_path}: {exc}"
            ) from exc

        if not vocab:
            return {}

        first_key = next(iter(vocab.keys()))
        if isinstance(first_key, str):
            id_to_token = {int(v): k for k, v in vocab.items()}
        else:
            id_to_token = {int(k): v for k, v in vocab.items()}

        return {
            tid: tok.replace("Ġ", " ").replace("Ċ", "\n")
            for tid, tok in id_to_token.items()
        }

    def encode(self, text: str) -> List[int]:
        """Tokenise text and return a flat list of integer token ids.

        Args:
            text: The text to encode.

        Returns:
            List of token ids.

        Raises:
            RuntimeError: If the LLM tokenizer fails.
        """
        try:
            raw_tokens = self.llm.encode(text)
        except Exception as exc:
            raise RuntimeError(f"Tokenization failed: {exc}") from exc

        if hasattr(raw_tokens, "tolist"):
            raw_tokens = raw_tokens.tolist()
        if (
            isinstance(raw_tokens, list)
            and raw_tokens
            and isinstance(raw_tokens[0], list)
        ):
            return [int(idx) for idx in raw_tokens[0]]
        return [int(idx) for idx in raw_tokens]

    def _get_valid_tokens(self, prefix: str, allowed: List[str]) -> List[int]:
        """Return tokens that keep *prefix* on a path to an allowed completion."""
        valid: List[int] = []
        for tid, tok in self.vocab_map.items():
            candidate = (prefix + tok).lstrip()
            for comp in allowed:
                if comp.startswith(candidate) or candidate.startswith(comp):
                    valid.append(tid)
                    break
        return valid

    def _generate_constrained(
        self,
        context_tokens: List[int],
        allowed_completions: List[str],
        max_len: int = 20,
    ) -> str:
        """Greedy constrained decoding.

        Args:
            context_tokens: Input token ids.
            allowed_completions: List of valid completion strings.
            max_len: Maximum tokens to generate.

        Returns:
            The generated completion string.

        Raises:
            RuntimeError: If the LLM fails to produce logits.
        """
        current = ""
        tokens = list(context_tokens)
        for _ in range(max_len):
            try:
                logits = self.llm.get_logits_from_input_ids(tokens)
            except Exception as exc:
                raise RuntimeError(f"LLM inference failed: {exc}") from exc

            logits_arr = np.array(logits, dtype=np.float32)
            valid = self._get_valid_tokens(current, allowed_completions)
            if not valid:
                break

            mask = np.full_like(logits_arr, -np.inf)
            for tid in valid:
                mask[tid] = logits_arr[tid]

            chosen = int(np.argmax(mask))
            tok = self.vocab_map.get(chosen, "")
            current += tok
            tokens.append(chosen)

            clean = current.lstrip()
            if clean in allowed_completions:
                return clean

        return allowed_completions[0] if allowed_completions else ""

    def _generate_function_name(self, user_prompt: str) -> str:
        """Build the selection prompt and run constrained generation."""
        prompt = (
            f'User Request: "{user_prompt}"\n'
            f"Which function should be called?\n"
        )
        for name, spec in self.func_map.items():
            prompt += f"- {name}: {spec.get('description', '')}\n"
        prompt += "Function Name: "
        tokens = self.encode(prompt)
        return self._generate_constrained(tokens, self.func_names, max_len=15)

    # ------------------------------------------------------------------
    # Deterministic parameter extraction (no LLM)
    # ------------------------------------------------------------------

    def _extract_numbers(self, text: str) -> List[float]:
        """Scan text for numeric literals (integers, negatives, decimals)."""
        numbers: List[float] = []
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch == "-" or ch.isdigit():
                j = i
                if text[j] == "-":
                    j += 1
                if j < n and text[j].isdigit():
                    num_str = ""
                    if text[i] == "-":
                        num_str = "-"
                        i += 1
                    while i < n and text[i].isdigit():
                        num_str += text[i]
                        i += 1
                    if i < n and text[i] == ".":
                        num_str += "."
                        i += 1
                        while i < n and text[i].isdigit():
                            num_str += text[i]
                            i += 1
                    try:
                        numbers.append(float(num_str))
                    except ValueError:
                        pass
                    continue
            i += 1
        return numbers

    def _extract_quoted_strings(self, text: str) -> List[str]:
        """Extract quoted substrings, respecting backslash-escaped quotes."""
        strings: List[str] = []
        i, n = 0, len(text)
        while i < n:
            if text[i] in ('"', "'"):
                quote = text[i]
                i += 1
                start = i
                while i < n:
                    if text[i] == "\\" and i + 1 < n and text[i + 1] == quote:
                        i += 2
                    elif text[i] == quote:
                        strings.append(text[start:i])
                        i += 1
                        break
                    else:
                        i += 1
                continue
            i += 1
        return strings

    def _extract_path_like(self, text: str) -> Optional[str]:
        """Scan text for unquoted filesystem paths (Unix or Windows)."""
        i, n = 0, len(text)
        while i < n:
            ch = text[i]
            if ch in ("/", "~"):
                start = i
                i += 1
                while i < n and text[i] not in " \t\n\"\'":
                    i += 1
                return text[start:i]
            if (
                i + 2 < n
                and text[i + 1] == ":"
                and text[i + 2] == "\\"
                and text[i].isalpha()
            ):
                start = i
                i += 3
                while i < n and text[i] not in " \t\n\"\'":
                    i += 1
                return text[start:i]
            i += 1
        return None

    def _extract_quoted_after_keyword(
        self, prompt: str, keywords: List[str]
    ) -> Optional[str]:
        """Find the first quoted string after any of the given keywords."""
        low = prompt.lower()
        for kw in keywords:
            idx = low.find(kw)
            if idx == -1:
                continue
            rest = prompt[idx + len(kw):]
            for quote in ("'", '"'):
                start = rest.find(quote)
                if start == -1:
                    continue
                j = start + 1
                while j < len(rest):
                    if (
                        rest[j] == "\\"
                        and j + 1 < len(rest)
                        and rest[j + 1] == quote
                    ):
                        j += 2
                    elif rest[j] == quote:
                        return rest[start + 1:j]
                    else:
                        j += 1
        return None

    def _extract_by_param_name(
        self, prompt: str, param_name: str
    ) -> Optional[str]:
        """Extract a value using the parameter name as a lookup key.

        Strategies (in order):
          1. Colon-after    : "param_name: value"
          2. Quoted-after   : first quoted string after param_name
          3. Word-before    : word immediately preceding param_name
          4. Contextual-kw  : e.g. "replace" → look for "with '...'"
        """
        low = prompt.lower()
        name_low = param_name.lower()

        if len(name_low) >= 2:
            colon_idx = low.find(name_low + ":")
            if colon_idx != -1:
                return prompt[colon_idx + len(name_low) + 1:].strip()

            idx = low.find(name_low)
            if idx != -1:
                rest = prompt[idx + len(param_name):]
                for quote in ("'", '"'):
                    start = rest.find(quote)
                    if start != -1:
                        j = start + 1
                        while j < len(rest):
                            if (
                                rest[j] == "\\"
                                and j + 1 < len(rest)
                                and rest[j + 1] == quote
                            ):
                                j += 2
                            elif rest[j] == quote:
                                return rest[start + 1:j]
                            else:
                                j += 1

                before = prompt[:idx].rstrip()
                if before:
                    words = before.split()
                    if words:
                        word = words[-1]
                        if word.lower() in self._ARTICLES and len(words) > 1:
                            word = words[-2]
                        return word

        for hint_key, keywords in self.PARAM_HINTS.items():
            if hint_key in name_low:
                val = self._extract_quoted_after_keyword(prompt, keywords)
                if val is not None:
                    return val

        return None

    def _infer_regex(self, prompt: str) -> Optional[str]:
        """Look for known natural-language regex hints."""
        low = prompt.lower()
        for phrase, regex in self.REGEX_INFERENCE.items():
            if phrase in low:
                return regex
        return None

    def _maybe_symbol(self, word: str) -> str:
        """Map spoken words like "asterisks" to their symbol."""
        return self.WORD_TO_SYMBOL.get(word.lower(), word)

    def _get_properties(self, func_def: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise a function definition into a flat param_name → schema map."""
        params = func_def.get("parameters", {})
        if not isinstance(params, dict):
            return {}
        if "properties" in params:
            return params["properties"]
        if any(isinstance(v, dict) and "type" in v for v in params.values()):
            return params
        return {}

    def run_prompt(self, user_prompt: str) -> Dict[str, Any]:
        """Convert a natural-language prompt into a structured call dict.

        This method is fully defensive: every extraction step is isolated so
        that a failure in one parameter never crashes the whole pipeline.

        Returns:
            {"prompt": ..., "name": ..., "parameters": {...}}
        """
        if not self.func_map:
            return {"prompt": user_prompt, "name": "", "parameters": {}}

        # --- Function name (single constrained LLM call) ---
        try:
            func_name = self._generate_function_name(user_prompt)
        except Exception:
            func_name = self.func_names[0] if self.func_names else ""

        func_def = self.func_map.get(func_name, {})
        properties = self._get_properties(func_def)

        # --- Pre-scan prompt data ---
        try:
            numbers = self._extract_numbers(user_prompt)
        except Exception:
            numbers = []
        try:
            quoted_strings = self._extract_quoted_strings(user_prompt)
        except Exception:
            quoted_strings = []

        prompt_lower = user_prompt.lower()
        num_idx = 0
        str_idx = 0
        regex_inferred: Optional[str] = None
        params: Dict[str, Any] = {}

        # --- Extract each parameter independently ---
        for pname, pschema in properties.items():
            try:
                ptype = pschema.get("type", "string")
                enum_vals = pschema.get("enum", [])

                if enum_vals:
                    chosen = None
                    for v in enum_vals:
                        if v.lower() in prompt_lower:
                            chosen = v
                            break
                    params[pname] = chosen if chosen is not None else enum_vals[0]
                    continue

                if ptype == "boolean":
                    params[pname] = (
                        "true" in prompt_lower or "yes" in prompt_lower
                    )
                    continue

                if ptype in ("integer", "number"):
                    if num_idx < len(numbers):
                        val = numbers[num_idx]
                        num_idx += 1
                        params[pname] = int(val) if ptype == "integer" else val
                    else:
                        params[pname] = 0 if ptype == "integer" else 0.0
                    continue

                if ptype == "string":
                    if any(kw in pname.lower() for kw in ("regex", "pattern")):
                        if regex_inferred is None:
                            regex_inferred = self._infer_regex(user_prompt)
                        if regex_inferred:
                            params[pname] = regex_inferred
                            continue

                    value = self._extract_by_param_name(user_prompt, pname)
                    if value is not None:
                        if value in quoted_strings:
                            quoted_strings.remove(value)
                        params[pname] = self._maybe_symbol(value)
                        continue

                    if any(kw in pname.lower() for kw in ("path", "file")):
                        path_val = self._extract_path_like(user_prompt)
                        if path_val is not None:
                            params[pname] = path_val
                            continue

                    if str_idx < len(quoted_strings):
                        params[pname] = quoted_strings[str_idx]
                        str_idx += 1
                        continue

                    words = user_prompt.split()
                    for w in reversed(words):
                        clean = w.strip(".,!?;:'\"()[]{}")
                        if clean and not any(ch.isdigit() for ch in clean):
                            params[pname] = self._maybe_symbol(clean)
                            break
                    else:
                        params[pname] = ""
                    continue

                params[pname] = None
            except Exception:
                # One parameter failing should never kill the whole call.
                params[pname] = None

        return {
            "prompt": user_prompt,
            "name": func_name,
            "parameters": params,
        }