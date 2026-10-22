"""Compact reference implementation of IndicTrans2 (en -> indic) for Project 17.

Mirrors the official modeling_indictrans.py (fairseq-style pre-norm transformer, sinusoidal
positions starting at id 2, scaled embeddings, layernorm_embedding, final layer norms, GELU), so
the C engine on the STM32N6 can be checked against plain, readable math. Loads model.safetensors
directly; no trust_remote_code.

    from it2_ref import IT2
    m = IT2(model_dir)                    # or IT2(model_dir, dec_layers=[0, 9, 17]) for a shrunk decoder
    m.translate(["Hello, how are you?"])  # -> ["नमस्ते, आप कैसे हैं?"]
"""
import json
import math
import os
import re

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

D, HEADS, PAD, BOS, EOS = 512, 8, 1, 0, 2


def sinusoids(n, dim=D, pad=PAD):
    half = dim // 2
    step = math.log(10000) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float) * -step)
    ang = torch.arange(n, dtype=torch.float)[:, None] * freq[None, :]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    emb[pad] = 0
    return emb


POS = sinusoids(512)


def restricted_spm(model_file, keep):
    """SentencePiece model in which every normal piece for which keep(piece) is false is UNUSED.
    Its BPE then splits such pieces back into the pieces they were merged from (what the removed
    SetVocabulary() did), like it2.c with a --src-keep blob."""
    import sentencepiece as spm
    from sentencepiece import sentencepiece_model_pb2 as pb
    proto = pb.ModelProto()
    proto.ParseFromString(open(model_file, "rb").read())
    for piece in proto.pieces:
        if piece.type == 1 and len(piece.piece) > 1 and not keep(piece.piece):
            piece.type = 5  # UNUSED
    return spm.SentencePieceProcessor(model_proto=proto.SerializeToString())


def ln(x, w, b):
    return F.layer_norm(x, (D,), w, b, 1e-5)


def lin(x, w, b):
    return F.linear(x, w, b)


def attend(q, k, v, mask=None):
    """q [T,D], k/v [S,D] -> [T,D]; mask [T,S] additive."""
    hd = D // HEADS
    q = q.view(q.shape[0], HEADS, hd).transpose(0, 1)
    k = k.view(k.shape[0], HEADS, hd).transpose(0, 1)
    v = v.view(v.shape[0], HEADS, hd).transpose(0, 1)
    s = q @ k.transpose(1, 2)
    if mask is not None:
        s = s + mask
    return (torch.softmax(s, dim=-1) @ v).transpose(0, 1).reshape(-1, D)


class IT2:
    def __init__(self, model_dir, dec_layers=None, weights=None):
        self.dir = model_dir
        self.w = weights if weights is not None else load_file(os.path.join(model_dir, "model.safetensors"))
        self.n_enc = len({k.split(".")[3] for k in self.w if k.startswith("model.encoder.layers.")})
        n_dec = len({k.split(".")[3] for k in self.w if k.startswith("model.decoder.layers.")})
        self.dec_layers = list(range(n_dec)) if dec_layers is None else list(dec_layers)
        self.src_vocab = json.load(open(os.path.join(model_dir, "dict.SRC.json"), encoding="utf-8"))
        self.tgt_vocab = json.load(open(os.path.join(model_dir, "dict.TGT.json"), encoding="utf-8"))
        self.tgt_inv = {v: k for k, v in self.tgt_vocab.items()}
        import sentencepiece as spm
        self.src_spm = spm.SentencePieceProcessor(model_file=os.path.join(model_dir, "model.SRC"))
        from sacremoses import MosesPunctNormalizer, MosesTokenizer
        self.norm, self.mtok = MosesPunctNormalizer(lang="en"), MosesTokenizer(lang="en")
        self.lm_head = self.w.get("lm_head.weight", self.w["model.decoder.embed_tokens.weight"])

    # ---- text <-> ids -------------------------------------------------------------------------
    def preprocess(self, text):
        return self.mtok.tokenize(self.norm.normalize(text.strip()), escape=False, return_str=True)

    def encode_text(self, text):
        pieces = ["eng_Latn", "hin_Deva"] + self.src_spm.EncodeAsPieces(self.preprocess(text))
        unk = self.src_vocab["<unk>"]
        return [self.src_vocab.get(p, unk) for p in pieces] + [EOS]

    def decode_ids(self, ids):
        pieces = [self.tgt_inv.get(i, "") for i in ids if i not in (BOS, PAD, EOS)]
        text = "".join(pieces).replace("▁", " ").strip()
        # Indic trivial detokenizer: no space before punctuation.
        return re.sub(r"\s+([,.?!।;:])", r"\1", text)

    # ---- network ------------------------------------------------------------------------------
    def encode(self, ids):
        w, p = self.w, "model.encoder."
        x = F.embedding(torch.tensor(ids), w[p + "embed_tokens.weight"]) * math.sqrt(D)
        x = x + POS[2:2 + len(ids)]
        x = ln(x, w[p + "layernorm_embedding.weight"], w[p + "layernorm_embedding.bias"])
        for i in range(self.n_enc):
            q = p + "layers.%d." % i
            h = ln(x, w[q + "self_attn_layer_norm.weight"], w[q + "self_attn_layer_norm.bias"])
            a = attend(lin(h, w[q + "self_attn.q_proj.weight"], w[q + "self_attn.q_proj.bias"]) / math.sqrt(D // HEADS),
                       lin(h, w[q + "self_attn.k_proj.weight"], w[q + "self_attn.k_proj.bias"]),
                       lin(h, w[q + "self_attn.v_proj.weight"], w[q + "self_attn.v_proj.bias"]))
            x = x + lin(a, w[q + "self_attn.out_proj.weight"], w[q + "self_attn.out_proj.bias"])
            h = ln(x, w[q + "final_layer_norm.weight"], w[q + "final_layer_norm.bias"])
            x = x + lin(F.gelu(lin(h, w[q + "fc1.weight"], w[q + "fc1.bias"])), w[q + "fc2.weight"], w[q + "fc2.bias"])
        return ln(x, w[p + "layer_norm.weight"], w[p + "layer_norm.bias"])

    def cross_kv(self, enc):
        w, out = self.w, []
        for i in self.dec_layers:
            q = "model.decoder.layers.%d.encoder_attn." % i
            out.append((lin(enc, w[q + "k_proj.weight"], w[q + "k_proj.bias"]),
                        lin(enc, w[q + "v_proj.weight"], w[q + "v_proj.bias"])))
        return out

    def step(self, token, pos, cache, xkv):
        """One decoder step. cache: per layer [K list, V list] of self-attention rows."""
        w, p = self.w, "model.decoder."
        x = F.embedding(torch.tensor([token]), w[p + "embed_tokens.weight"]) * math.sqrt(D)
        x = x + POS[2 + pos:3 + pos]
        x = ln(x, w[p + "layernorm_embedding.weight"], w[p + "layernorm_embedding.bias"])
        for li, i in enumerate(self.dec_layers):
            q = p + "layers.%d." % i
            h = ln(x, w[q + "self_attn_layer_norm.weight"], w[q + "self_attn_layer_norm.bias"])
            k = lin(h, w[q + "self_attn.k_proj.weight"], w[q + "self_attn.k_proj.bias"])
            v = lin(h, w[q + "self_attn.v_proj.weight"], w[q + "self_attn.v_proj.bias"])
            cache[li][0] = torch.cat([cache[li][0], k]) if cache[li][0] is not None else k
            cache[li][1] = torch.cat([cache[li][1], v]) if cache[li][1] is not None else v
            a = attend(lin(h, w[q + "self_attn.q_proj.weight"], w[q + "self_attn.q_proj.bias"]) / math.sqrt(D // HEADS),
                       cache[li][0], cache[li][1])
            x = x + lin(a, w[q + "self_attn.out_proj.weight"], w[q + "self_attn.out_proj.bias"])
            h = ln(x, w[q + "encoder_attn_layer_norm.weight"], w[q + "encoder_attn_layer_norm.bias"])
            a = attend(lin(h, w[q + "encoder_attn.q_proj.weight"], w[q + "encoder_attn.q_proj.bias"]) / math.sqrt(D // HEADS),
                       xkv[li][0], xkv[li][1])
            x = x + lin(a, w[q + "encoder_attn.out_proj.weight"], w[q + "encoder_attn.out_proj.bias"])
            h = ln(x, w[q + "final_layer_norm.weight"], w[q + "final_layer_norm.bias"])
            x = x + lin(F.gelu(lin(h, w[q + "fc1.weight"], w[q + "fc1.bias"])), w[q + "fc2.weight"], w[q + "fc2.bias"])
        x = ln(x, w[p + "layer_norm.weight"], w[p + "layer_norm.bias"])
        return F.linear(x, self.lm_head)[0]

    @torch.no_grad()
    def greedy(self, src_ids, max_len=128, allowed=None):
        enc = self.encode(src_ids)
        xkv = self.cross_kv(enc)
        cache = [[None, None] for _ in self.dec_layers]
        tok, out = EOS, []  # decoder_start_token_id = 2
        for pos in range(max_len):
            logits = self.step(tok, pos, cache, xkv)
            if allowed is not None:
                logits = logits[allowed]
                tok = int(allowed[int(torch.argmax(logits))])
            else:
                tok = int(torch.argmax(logits))
            if tok == EOS:
                break
            out.append(tok)
        return out

    @torch.no_grad()
    def beam(self, src_ids, beams=4, max_len=128, len_pen=1.0):
        enc = self.encode(src_ids)
        xkv = self.cross_kv(enc)
        hyps = [([], 0.0, [[None, None] for _ in self.dec_layers], EOS)]
        done = []
        for pos in range(max_len):
            cand = []
            for toks, score, cache, last in hyps:
                lp = torch.log_softmax(self.step(last, pos, cache, xkv), dim=-1)
                top = torch.topk(lp, beams)
                for s, t in zip(top.values.tolist(), top.indices.tolist()):
                    cand.append((toks + [t], score + s, cache, t))
            cand.sort(key=lambda c: c[1], reverse=True)
            hyps = []
            for toks, score, cache, t in cand:
                if t == EOS:
                    done.append((toks[:-1], score / (len(toks) ** len_pen)))
                elif len(hyps) < beams:
                    hyps.append((toks, score, [[c[0].clone(), c[1].clone()] for c in cache], t))
                if len(hyps) == beams:
                    break
            if len(done) >= beams or not hyps:
                break
        if not done:
            done = [(h[0], h[1] / max(1, len(h[0])) ** len_pen) for h in hyps]
        return max(done, key=lambda d: d[1])[0]

    def translate(self, sentences, beams=1):
        out = []
        for s in sentences:
            ids = self.encode_text(s)
            out.append(self.decode_ids(self.greedy(ids) if beams == 1 else self.beam(ids, beams)))
        return out
