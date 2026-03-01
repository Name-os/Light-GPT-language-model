from settings import *
from bpe import *


# creates a dict mapping idx to its byte version, or idx : bytes(idx)
vocab = {idx : bytes(idx) for idx in range(256)}
# now add the merges by making idx, where the merge is, into pair_0 + pair_1
for (pair_0, pair_1), idx in merges.items():
    vocab[idx] = vocab[pair_0] + vocab[pair_1]

def decode(num_list):
    # get the raw bytes for each idx and turn into tokens
    tokens = b"".join(vocab[idx] for idx in num_list)
    # decode the tokens
    # "replace" means if there is invalid bytes, replace with placeholder instead of error
    text = tokens.decode("utf-8", errors="replace")
    # return text
    return text

def encode(text):
    # get a list of int bytes
    tokens = list(text.encode("utf-8"))

    # merge the tokens using merges so bpe is used
    while True:
        # get stats on token pairs and only care about token num after merging
        stats = get_stats(tokens)
        # get the best pair to merge which is the one that ocurs first, or lowest num
        # use min on stats and key based on token num, if it doesnt exist, replace with inf so not in min
        pair = min(stats, key=lambda pair: merges.get(pair, float("inf")))

        # check if there is anything to merge
        if pair not in merges:
            break
        # get the replace tokens for merge by looking up the pair
        idx = merges[pair]
        # now replace the pairs with the new token using merge()
        tokens = merge(tokens, pair, idx)

    return tokens
