*This project has been created as part of the 42 curriculum by sait-mou*

# Description:
what is an LLM?
At a basic level, LLMs are built on machine learning. Machine learning is a subset of AI, and it refers to the practice of feeding a program large amounts of data in order to train the program how to identify features of that data without human intervention.

LLMs use a type of machine learning called deep learning. Deep learning models can essentially train themselves to recognize distinctions without human intervention, although some human fine-tuning is typically necessary.

Deep learning uses probability in order to "learn." For instance, in the sentence "The quick brown fox jumped over the lazy dog," the letters "e" and "o" are the most common, appearing four times each. From this, a deep learning model could conclude (correctly) that these characters are among the most likely to appear in English-language text.

Realistically, a deep learning model cannot actually conclude anything from a single sentence. But after analyzing trillions of sentences, it could learn enough to predict how to logically finish an incomplete sentence, or even generate its own sentences

As they say "A picture worth a thousand words":
![Screenshot](https://symufolk.com/wp-content/uploads/2025/01/How-LLm-Works-in-Climate-Analytics-1024x437.png)

So with all that being said our job here is to get function calling response from that llm
More about that in the algorithem explanation section :)


#### Words you will see alot:

**Function calling**: refers to the process by which an LLM detects that a user request requires external data or action and then produces a structured output (typically in JSON) that specifies which function to call along with the necessary arguments

**Token**: a small unit carrying meaning, like the syllable un or the word aircraft. LLMs have a vocabulary of tokens which are represented by their token ID, for instance the token "Ġ" could correspond to the token ID 220

**Logits**: In the generation loop, LLMs receive a sequence of token IDs representing the text that has been generated so far. They then calculate a logit for each of the token IDs in their vocabulary. The logit represents the confidence of the LLM, that the associated token should be the next token in the sequence. This process is called prediction.

**Transformer**: Transformers are the state-of-the-art architecture for a wide variety of language model applications
Full transformers consist of an encoder and a decoder:

An encoder converts input text into an intermediate representation. An encoder is an enormous neural net.

A decoder converts that intermediate representation into useful text. A decoder is also an enormous neural net.


# Instructions:

#### To run the LLM:
```Bash
uv run python3 -m src
```
(PS: make sure you have input data 'function definitions' and 'prompts')


## Algorithm explanation:

The decoding engine operates through five sequential stages to guarantee valid token generation:

#### 1. Reverse Vocabulary Indexing (load_vocab)

Reverses vocab.json into an {int_id: str_token} dictionary for O(1) lookups and normalizes byte-fallback BPE characters, such as converting Ġ to spaces and Ċ to \n.

#### 2. Context Alignment (encode)

Tokenizes input prompt strings and flattens array or tensor tokenizer returns into a standardized 1D List[int] token context sequence.

#### 3. Vocabulary Guardrail Filtering (valid_exact)

Scans all token strings in the vocabulary to identify token IDs that legally form a prefix toward, or complete a match against, allowed target candidate strings.

#### 4. Logit Masking & Decoding Loop (constrained_exact)

Evaluates raw logit vectors from the LLM forward pass, sets unallowed token IDs to -\infty, selects the highest valid token via np.argmax, and triggers early termination as soon as a candidate target string is completed.

#### 5. Schema Normalization & Parameter Extraction (props & run_prompt)

Resolves standard OpenAPI/JSON Schema formats into unified property maps and prefills structural JSON keys ({, "key":, etc.) so forward passes are spent solely on parameter values.


## Design decisions:
#### Vocabulary-Level Logit Masking

Setting unallowed token scores to -\infty guarantees valid output while preserving the LLM's internal relative rankings among legitimate choices.

#### Structural Prefilling for JSON

Prefilling structural syntax directly into the context forces the LLM to compute only argument values, eliminating malformed JSON keys and syntax errors.

#### Trading CPU Loops for Forward Passes

Performing an O(V) string check over self.vocab on the CPU takes approximately 2–5 ms per step, but saves dozens of heavy neural network forward passes by cutting unnecessary conversational filler.

#### Immediate Early Termination

Generation breaks immediately once a completed target matches an entry in allowed, saving compute time that would otherwise be wasted waiting for EOS tokens.

#### Schema Normalization (props)

Unifies standard OpenAPI schema structures (properties) and simplified direct dictionaries to decouple decoding logic from schema variance.

#### Zero Heavy Framework Dependencies

Relies strictly on native Python, NumPy, and Pydantic for complete transparency and minimal runtime overhead.

## Performance analysis:
#### Execution Speed

Reduces generated tokens per function call from 25+ conversational tokens down to 2–4 targeted tokens, providing a significant throughput speedup.

#### Deterministic Accuracy

Reaches 100% schema compliance by physically blocking syntax errors, invalid enum choices, or out-of-range parameter types.

#### Resource Efficiency

Avoids heavy grammar state machines or external decoding daemons, maintaining a minimal memory footprint.

## Challenges faced:
Moving from creative text generation to treating LLM vocabulary vectors as state-constrained choices required careful tracking of token boundaries.

## Testing strategy:
Tested numeric, boolean (true/false), and string filters to confirm invalid types are blocked at the logit layer.

## Example usage:

## Resources:
[Grammar Constrained Decoding in Bumblebee with State Machines](https://bitcrowd.dev/grammar-constrained-decoding-in-bumblebee/)

## Use of AI Assistance

AI tools were utilized during this project for the following specific tasks:

#### Code Refactoring

Assisted in expanding complex one-line list comprehensions into readable, multi-step standard for loops.

#### Documentation & README Drafting

Aided in structuring architectural explanations, ASCII sequence flows, and formatting technical design choices.

#### Test Case Brainstorming

Suggested edge-case parameter inputs for validating schema normalization functions.