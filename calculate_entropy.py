# requirements.txt:
# numpy
# tqdm
# huggingface_hub
# datasets==3.6.0
# transformers
# clickhouse_connect

HUGGINGFACE_TOKEN=""
CLICKHOUSE_USER=""
CLICKHOUSE_PASSWORD=""
CLICKHOUSE_HOST=""
RANDOM_SEED = 42

import random
import math
import numpy as np
from tqdm import tqdm
from huggingface_hub import login
from datasets import load_dataset
from transformers import AutoTokenizer
import clickhouse_connect

random.seed(RANDOM_SEED)

client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        user=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        secure=True,
    )


# Log into HF
login(token=HUGGINGFACE_TOKEN)

# Load dataset in streaming mode
ds_stream = load_dataset("carolina-c4ai/corpus-carolina", split="corpus", streaming=True, trust_remote_code=True)

# Load the SentencePiece tokenizer
model_path = "TucanoBR/ViTucano-1b5-v1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
VOCAB_SIZE = tokenizer.vocab_size

qry_unigram = client.query("SELECT t1, sum(count) from unigrams group by t1 ").result_set
qry_bigram= client.query("SELECT t1, t2, sum(count) from bigrams group by t1,t2").result_set
qry_trigram = client.query("SELECT t1,t2,t3, sum(count) from trigrams group by t1,t2,t3").result_set

tot_unigram = client.query("SELECT sum(count) from unigrams").result_set[0][0]
tot_bigram = client.query("SELECT sum(count) from bigrams").result_set[0][0]

unigrams = {}
bigrams = {}
trigrams = {}

for t1, count in qry_unigram:
  unigrams[t1] = count

for t1, t2, count in qry_bigram:
  bigrams[(t1,t2)] = count

for t1, t2, t3, count in qry_trigram:
  trigrams[(t1,t2,t3)] = count

total_prob = 0
total_chars = 0

for test_sample in tqdm(ds_stream, total= 2108999 ):
  p = random.random()

  if p > 0.1:
    continue

  test_sample = test_sample['text']
  total_chars += len(test_sample)
  encoding = tokenizer.encode(test_sample)

  if not encoding:
    continue

  t1 = encoding[0]
  # P(t1)
  total_prob -= (np.log2(unigrams.get(t1,0)+1)-np.log2(tot_unigram+VOCAB_SIZE))

  if len(encoding) < 2:
    continue

  t2 = encoding[1]
  # P(t2 | t1)
  total_prob -= (np.log2(bigrams.get((t1,t2), 0)+1)-np.log2(unigrams.get(t1,0)+VOCAB_SIZE))

  # P(t3 | t1, t2)
  for t1,t2,t3 in zip(encoding, encoding[1:], encoding[2:]):
    total_prob -= (np.log2(trigrams.get((t1,t2,t3),0)+1)-np.log2(bigrams.get((t1,t2),0)+VOCAB_SIZE))

print(f"Total entropy rate (bits): {total_prob/total_chars}")

