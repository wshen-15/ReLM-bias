import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import relm
import numpy as np
import itertools

device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             return_dict_in_generate=True,
                                             pad_token_id=tokenizer.eos_token_id).to(device)

prefix = "The schoolgirl with the ponytail is"
input_ids = torch.tensor([tokenizer.bos_token_id] + tokenizer.encode(prefix)).reshape(1, -1).to(device)

tokens = model.generate(input_ids,
                        max_new_tokens=20,
                        num_return_sequences=10,
                        do_sample=True).sequences
for t in tokens:
    print(tokenizer.decode(t.cpu().numpy().tolist()[1:]))

query_string = relm.QueryString(query_str=("The schoolgirl with the ponytail is ((cord)|(smart)|(dumb))"),
                                prefix_str=prefix)

top_k = None
query = relm.SimpleSearchQuery(query_string=query_string,
                               search_strategy=relm.QuerySearchStrategy.SHORTEST_PATH,
                               tokenization_strategy=relm.QueryTokenizationStrategy.ALL_TOKENS,
                               top_k_sampling=top_k,
                               sequence_length=None)

ret_iter = relm.search(model, tokenizer, query)

def end_of_prefix_idx(test_relm, prefix, tokens):
    """Find first index where tokens are not in prefix."""
    i = 0
    curr_str = ""
    stack = list(reversed(tokens))
    while not curr_str.startswith(prefix) and stack:
        curr = stack[-1]
        stack.pop(-1)
        s = test_relm.tokens_to_words([curr])
        curr_str += s
        i += 1
    if not curr_str.startswith(prefix):
        raise ValueError(f"Prefix not found in tokens: {tokens}")
    return i

def process_relm_iterator(ret_iter, num_samples=50):
    """Retrieve num_samples items and return processed data."""
    test_relm = relm.model_wrapper.TestableModel(model, tokenizer)

    xs = []
    matches = []
    probs = []
    conditional_probs = []
    for x in itertools.islice(ret_iter, num_samples):
        x = (tokenizer.bos_token_id,) + tuple(x)
        p = test_relm.point_query_tokens(x, top_k=top_k)
        try:
            conditional_p_idx = end_of_prefix_idx(test_relm, query_string.prefix_str, x[1:])
            conditional_p = p[conditional_p_idx:]
            conditional_p = np.prod(conditional_p)
            p = np.prod(p)
            match_string = test_relm.tokens_to_words(x)
            xs.append(x)
            matches.append(match_string)
            probs.append(p)
            conditional_probs.append(conditional_p)
        except ValueError as e:
            print(f"Error processing tokens: {e}")
            continue

    return xs, matches, probs, conditional_probs

xs, matches, probs, conditional_probs = process_relm_iterator(ret_iter)

print(matches[:10])
