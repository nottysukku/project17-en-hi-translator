"""Check it2_ref against the official IndicTrans2 code (greedy, no KV cache) and time it."""
import sys
import time

import torch

sys.stdout.reconfigure(encoding="utf-8")
from transformers import AutoModelForSeq2SeqLM

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from it2_ref import IT2, EOS  # noqa: E402

MODEL = sys.argv[1]
ref = IT2(MODEL)
off = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).eval()
for s in ["Hello, how are you?", "I don't understand what you are saying."]:
    ids = ref.encode_text(s)
    t = time.time(); mine = ref.greedy(ids); t1 = time.time() - t
    with torch.no_grad():
        enc_off = off.model.encoder(input_ids=torch.tensor([ids])).last_hidden_state[0]
        dec = [EOS]
        for _ in range(len(mine) + 1):
            out = off(input_ids=torch.tensor([ids]), decoder_input_ids=torch.tensor([dec]), use_cache=False)
            tok = int(out.logits[0, -1].argmax())
            if tok == EOS:
                break
            dec.append(tok)
    enc_err = float((ref.encode(ids) - enc_off).abs().max())
    print("%s\n  mine: %s\n  off : %s\n  same=%s enc_maxerr=%.2e  ref %.2fs" %
          (s, ref.decode_ids(mine), ref.decode_ids(dec[1:]), mine == dec[1:], enc_err, t1))
