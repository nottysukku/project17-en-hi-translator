"""Check the C TTS front end (tts_front_host) against the Python reference and VitsModel.

    python tts_front_check.py <mms_dir> <tts_front.bin> <tts_front_host.exe> [seed] [--float-activations]

--float-activations: for a host built with -DTTS_FRONT_FLOAT_ACTIVATIONS (int8 weights, float
inputs); the reference then skips the activation rounding too, so the two must agree to float
rounding. This checks the structure; the int8 activation rounding (as in it2.c) is checked by
the normal run.

For each test sentence, with the same noise (tts_front.c's generator, reproduced here):
  ids        C tokenizer vs VitsTokenizer
  C vs ref   tts_front_ref.Front with the blob's precision (int8 text encoder, float16 duration
             predictor): durations, logw, prior means/log-variances and latents should agree to
             float rounding
  C vs VITS  the float model (VitsModel): durations and latents, to show what int8 costs
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
from transformers import AutoTokenizer, VitsModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_front_ref import Front, text_to_ids  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
MAX_IDS, CH = 511, 192
SENTS = ["नमस्ते, आप कैसे हैं?", "आपका नाम क्या है?", "मैं कल बाज़ार जा रहा हूँ।",
         "क्या आप रेलवे स्टेशन खोजने में मेरी मदद कर सकते हैं?", "मुझे समझ नहीं आ रहा है कि आप क्या कह रहे हैं।",
         "कमरे से बाहर निकलने पर कृपया बत्तियाँ बंद कर दें।", "इस पर कितना खर्च आता है?",
         "आपकी मदद के लिए बहुत-बहुत धन्यवाद।", "मेरा भाई बैंगलोर में सॉफ्टवेयर इंजीनियर के रूप में काम करता है।",
         "  आज मौसम बहुत अच्छा है, चलो टहलने चलते हैं। ", "डॉक्टर ने कहा कि मुझे आराम करना चाहिए।",
         "ट्रेन 10 मिनट देर से आएगी।"]


def mix32(x):
    x = np.asarray(x, np.uint32)
    with np.errstate(over="ignore"):
        x = x ^ (x >> np.uint32(16))
        x = x * np.uint32(0x21F0AAAD)
        x = x ^ (x >> np.uint32(15))
        x = x * np.uint32(0x735A2D97)
        x = x ^ (x >> np.uint32(15))
    return x


def gauss(seed, stream, index):
    """tts_front_gauss() for an array of indices, in float32 like the C code."""
    index = np.asarray(index, np.uint64)
    key = mix32(np.uint32((seed ^ (stream * 0x9E3779B9)) & 0xFFFFFFFF))
    a = mix32(key ^ ((2 * index) & 0xFFFFFFFF).astype(np.uint32))
    b = mix32(key ^ ((2 * index + 1) & 0xFFFFFFFF).astype(np.uint32))
    u1 = ((a >> np.uint32(8)).astype(np.float32) + np.float32(1)) * np.float32(1 / 16777216)
    u2 = (b >> np.uint32(8)).astype(np.float32) * np.float32(1 / 16777216)
    return np.sqrt(np.float32(-2) * np.log(u1)) * np.cos(np.float32(6.28318531) * u2)


def read_c(path):
    data = open(path, "rb").read()
    n, frames = struct.unpack_from("<II", data, 0)
    off = 8
    ids = np.frombuffer(data, np.uint16, n, off).tolist()
    off += 2 * n
    if frames == 0:
        return ids, None
    dur = np.frombuffer(data, np.uint16, n, off)
    off += 2 * n
    logw = np.frombuffer(data, np.float32, n, off)
    off += 4 * n
    stats = np.frombuffer(data, np.float32, n * 384, off).reshape(n, 384)
    off += 4 * n * 384
    z = np.frombuffer(data, np.float32, CH * frames, off).reshape(CH, frames)
    return ids, dict(dur=dur, logw=logw, stats=stats, z=z)


def main():
    mms, blob, exe = sys.argv[1], sys.argv[2], sys.argv[3]
    args = [a for a in sys.argv[4:] if not a.startswith("--")]
    seed = int(args[0]) if args else 1234
    float_act = "--float-activations" in sys.argv
    model = VitsModel.from_pretrained(mms).eval()
    tok = AutoTokenizer.from_pretrained(mms)
    vocab = json.load(open(os.path.join(mms, "vocab.json"), encoding="utf-8"))
    sd = model.state_dict()
    sd16 = {k: (v.half().float() if k.startswith("duration_predictor") else v) for k, v in sd.items()}
    ref = Front(sd16, 8, quantize_activations=not float_act)
    with tempfile.TemporaryDirectory() as tmp:
        inp = os.path.join(tmp, "in.txt")
        open(inp, "w", encoding="utf-8", newline="\n").write("\n".join(SENTS) + "\n")
        out = subprocess.run([exe, blob, inp, str(seed), tmp], capture_output=True, text=True, check=True).stdout
        times = [int(float(l.split("\t")[2])) for l in out.splitlines()]
        worst = dict(logw=0.0, stats=0.0, z=0.0)
        dur_same = dur_total = 0
        print("%-40s %4s %6s | %-24s | %s" % ("sentence", "ids", "frames", "C vs reference (max |diff|)",
                                                "C vs float VitsModel"))
        for i, text in enumerate(SENTS):
            ids, c = read_c(os.path.join(tmp, "front_%02d.bin" % i))
            assert ids == text_to_ids(text, vocab), "C tokenizer differs from tts_front_ref"
            same_tok = ids == tok(text)["input_ids"]
            s = seed + i
            nd = torch.from_numpy(np.stack([gauss(s, 0, np.arange(len(ids))),
                                            gauss(s, 0, MAX_IDS + np.arange(len(ids)))]))

            def eps(frames, s=s):
                idx = np.arange(frames)[None, :] * CH + np.arange(CH)[:, None]
                return torch.from_numpy(gauss(s, 1, idx))

            dur, z, inter = ref.run(ids, nd, eps)
            stats = torch.cat([inter["m"], inter["logs"]], 1).numpy()
            dur_same += int((dur.numpy() == c["dur"]).sum())
            dur_total += len(ids)
            d_logw = float(np.abs(inter["logw"].numpy() - c["logw"]).max())
            d_stats = float(np.abs(stats - c["stats"]).max())
            d_z = float(np.abs(z.numpy() - c["z"]).max()) if z.shape[1] == c["z"].shape[1] else float("nan")
            worst = {k: max(worst[k], v) for k, v in (("logw", d_logw), ("stats", d_stats), ("z", d_z))}
            # float VitsModel with the same noise
            fl = Front(sd)
            fdur, fz, _ = fl.run(ids, nd, eps)
            n = min(fz.shape[1], c["z"].shape[1])
            snr = 10 * np.log10(float((fz[:, :n] ** 2).sum()) / max(float(((fz[:, :n].numpy() - c["z"][:, :n]) ** 2).sum()), 1e-20))
            print("%-40s %4d %6d | logw %.1e stats %.1e z %.1e | frames %4d, latent SNR %5.1f dB%s" %
                  (text[:40], len(ids), c["z"].shape[1], d_logw, d_stats, d_z, fz.shape[1], snr,
                   "" if same_tok else "  (ids differ from VitsTokenizer: digits/nukta handling)"))
        print("C vs reference: %d of %d durations identical; worst |diff| logw %.1e, stats %.1e, latents %.1e" %
              (dur_same, dur_total, worst["logw"], worst["stats"], worst["z"]))
        print("C time on this PC: %d ms per sentence on average" % (sum(times) / len(times)))


if __name__ == "__main__":
    main()
