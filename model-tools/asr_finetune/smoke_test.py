"""PC smoke test of the whole notebook path, on the CPU, with a stand-in network.

    python smoke_test.py <work_dir> <parquet with "audio" and "sentence" columns>

The stand-in has CynipsNet's layers (from the compile report) and random weights; it is saved
as a QDQ ONNX file like the board's cynipsnet_punct1.onnx, once with the input as [1, 64, 600]
and once as [1, 600, 64] plus a Transpose. Then, as the notebook does: load it into PyTorch,
check it computes what ONNX Runtime computes, train a few steps (the loss must fall), write the
float ONNX back (must match PyTorch), quantize to int8 with the board's input scale, decode with
the board simulation, and write network_shape.h.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import asr_lib as A  # noqa: E402


class ChannelsLast(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):  # [1, 600, 64] -> [1, 300, 32]
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def make_standin(work, windows, channels_last):
    torch.manual_seed(0)
    net = A.CynipsLike().eval()
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, torch.nn.Conv1d):
                m.weight.mul_(0.5)
    print("stand-in parameters: %d (the compile report says 7.2 M)" % A.count_params(net))
    model = ChannelsLast(net) if channels_last else net
    shape = (1, A.WIN_FRAMES, A.BANDS) if channels_last else (1, A.BANDS, A.WIN_FRAMES)
    fp = os.path.join(work, "standin_%s_float.onnx" % ("cl" if channels_last else "cf"))
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(model, torch.zeros(shape), fp, input_names=["x"], output_names=["logits"],
                          opset_version=17, dynamo=False)
    qdq = fp.replace("_float", "_qdq")
    A.quantize_int8(fp, qdq, windows)
    return fp, qdq


def main():
    work, parquet = sys.argv[1], sys.argv[2]
    os.makedirs(work, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 2)
    src = A.Source("dtu", "DTU54DL/common-accent", ["*.parquet"], "sentence", punct=True)
    units = A.index_file((parquet, src, None))
    print("index: %d row groups, %.2f h, %d utterances" % (len(units), sum(u[2] for u in units),
                                                           sum(len(u[3]) for u in units)))
    dev = A.dev_set(src, [parquet], 6, split_speakers=False)
    windows = A.calibration_windows(dev, 40)
    print("calibration windows:", len(windows))

    for channels_last in (False, True):
        t0 = time.time()
        fp, qdq = make_standin(work, windows, channels_last)
        net = A.OnnxNet(qdq)
        print("layout %s: input %s output %s, %d parameters" % ("[1,600,64]" if channels_last else "[1,64,600]",
                                                                 net.in_dims, net.out_dims, A.count_params(net)))
        # 1. PyTorch from the QDQ file vs ONNX Runtime on the float file (differ by weight rounding only)
        import onnxruntime as ort
        name, in_dims, out_dims = A.graph_layout(fp)
        w = windows[0]
        ref = A.output_to_steps(ort.InferenceSession(fp).run(None, {name: A.window_to_input(w, in_dims)})[0], out_dims)
        with torch.no_grad():
            got = net(torch.from_numpy(w.astype(np.float32) * A.INPUT_SCALE)[None])[0].numpy()
        cos = float((ref * got).sum() / np.linalg.norm(ref) / np.linalg.norm(got))
        print("  PyTorch (QDQ weights) vs ONNX Runtime (float): cosine %.4f, argmax agree %.1f%%"
              % (cos, 100 * float((ref.argmax(-1) == got.argmax(-1)).mean())))
        assert cos > 0.98

        # 2. a few training steps on one small batch: the loss must fall
        cfg = A.TrainConfig(lr=1e-3, warmup_steps=1, epochs=1, batch_seconds=40, kl_weight=0.5, amp=False,
                            spec_augment=False, norm_jitter=0.0)
        tr = A.Trainer(qdq, os.path.join(work, "ck"), cfg, torch.device("cpu"))
        tr.total_steps = 100
        ds = A.ShardStream({"dtu": units}, [src], 0, batch_seconds=40, max_batch=4, pool=12, augment_p=0.0)
        batch = next(iter(ds))
        b = list(batch)
        b[5] = torch.tensor([True, False] * (len(b[5]) // 2) + [True] * (len(b[5]) % 2))  # some replay rows
        b[4] = torch.tensor([True, False] * (len(b[4]) // 2) + [False] * (len(b[4]) % 2))  # some without marks
        losses = [tr.train_step(b)[0] for _ in range(12)]
        print("  CTC loss over 12 steps on one batch: %s" % " ".join("%.1f" % v for v in losses))
        assert losses[-1] < losses[0] * 0.97

        # 3. an augmented batch through the loader path
        ds2 = A.ShardStream({"dtu": units}, [src], 0, batch_seconds=40, max_batch=4, pool=12, augment_p=1.0)
        wave, lengths, targets, tl, punct, replay = next(iter(ds2))
        print("  augmented batch: %d rows, longest %.1f s" % (len(lengths), float(lengths.max()) / A.SR))

        # 4. checkpoint round trip and whole-utterance evaluation
        tr.save("last.pt")
        tr2 = A.Trainer(qdq, os.path.join(work, "ck"), cfg, torch.device("cpu"))
        assert tr2.load("last.pt") and tr2.step == tr.step
        res = tr.evaluate({"dtu_dev": dev[:3]})
        print("  evaluate:", {k: round(v["wer"], 3) for k, v in res.items()})

        # 5. float ONNX back out: must compute what PyTorch computes
        out_fp = os.path.join(work, "trained_float.onnx")
        tr.model.export_float_onnx(out_fp)
        name, in_dims, out_dims = A.graph_layout(out_fp)
        y = A.output_to_steps(ort.InferenceSession(out_fp).run(None, {name: A.window_to_input(w, in_dims)})[0],
                              out_dims)
        tr.model.eval()
        with torch.no_grad():
            g = tr.model(torch.from_numpy(w.astype(np.float32) * A.INPUT_SCALE)[None])[0].numpy()
        print("  exported float ONNX vs PyTorch: max difference %.2e" % float(np.abs(y - g).max()))
        assert np.abs(y - g).max() < 1e-3

        # 6. int8 with the board's input scale, board simulation, network_shape.h
        out_q = os.path.join(work, "trained_int8.onnx")
        in_s, out_s = A.quantize_int8(out_fp, out_q, windows)
        print("  int8: input scale %.10f (board %.10f), output scale %.6f" % (in_s, A.INPUT_SCALE, out_s))
        sim = A.BoardSim(out_q)
        text = sim.transcribe(dev[0][0])
        print("  board simulation decodes %r (random weights: any text)" % text[:60])
        A.write_network_shape(os.path.join(work, "network_shape.h"), "cynipsnet_inen1", in_s, out_s)

        # 7. the real loop for half a minute: two loader workers, logging, evaluation, stop and save
        if not channels_last:
            cfg = A.TrainConfig(lr=3e-4, warmup_steps=2, epochs=1, batch_seconds=40, max_batch=8, amp=False,
                                eval_every=6, log_every=2, session_hours=40 / 3600, workers=2)
            tr = A.Trainer(qdq, os.path.join(work, "ck_fit"), cfg, torch.device("cpu"))
            tr.fit({"dtu": units}, [src], {"dtu_dev": dev[:2]})
            assert tr.step > 0 and os.path.exists(os.path.join(work, "ck_fit", "last.pt"))
        print("  done in %.0f s" % (time.time() - t0))
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
