"""Check that the training features (asr_finetune/asr_lib.py) equal the board's (mel.c, norm.c).

    python asr_feature_check.py <feature_host> <parquet with an "audio" column> [n]

feature_host is built from firmware-n6/src/asr_engine/test/feature_host.c (build line in that
file). For n utterances it compares every log-mel frame and every int8 network window, and
prints the largest differences. Expected: log-mel within ~1e-4, window codes equal except for
a few that sit on a rounding edge (off by one).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "asr_finetune"))
import asr_lib as A  # noqa: E402


def main():
    exe, parquet = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    rows = pq.read_table(parquet).slice(0, n).to_pylist()
    worst_mel, codes, off1, more, wins = 0.0, 0, 0, 0, 0
    with tempfile.TemporaryDirectory() as tmp:
        for r in rows:
            pcm = A.decode_audio(r["audio"])
            pcm = np.concatenate([pcm, pcm, pcm])  # long enough for several windows
            pcm.tofile(os.path.join(tmp, "in.pcm"))
            subprocess.run([exe, os.path.join(tmp, "in.pcm"), os.path.join(tmp, "mel.f32"),
                            os.path.join(tmp, "win.i8")], check=True, stdout=subprocess.DEVNULL)
            c_mel = np.fromfile(os.path.join(tmp, "mel.f32"), np.float32).reshape(-1, A.BANDS)
            c_win = np.fromfile(os.path.join(tmp, "win.i8"), np.int8).reshape(-1, A.WIN_FRAMES, A.BANDS)
            py_mel = A.logmel_numpy(pcm)
            assert py_mel.shape == c_mel.shape, (py_mel.shape, c_mel.shape)
            worst_mel = max(worst_mel, float(np.abs(py_mel - c_mel).max()))
            py_win = np.stack([w for _, w in A.board_windows(c_mel)])
            assert py_win.shape == c_win.shape, (py_win.shape, c_win.shape)
            d = np.abs(py_win.astype(int) - c_win.astype(int))
            codes += d.size
            off1 += int((d == 1).sum())
            more += int((d > 1).sum())
            wins += len(c_win)
            # the same windows from the Python log-mel: how much the float differences matter
            py_win2 = np.stack([w for _, w in A.board_windows(py_mel)])
            d2 = np.abs(py_win2.astype(int) - c_win.astype(int))
            print("utterance %.1f s: %d frames, %d windows; log-mel max diff %.2e; codes from Python log-mel "
                  "differing by 1: %d of %d" % (len(pcm) / A.SR, len(c_mel), len(c_win),
                                                float(np.abs(py_mel - c_mel).max()), int((d2 > 0).sum()), d2.size))
    print("log-mel: largest difference %.2e" % worst_mel)
    print("norm.c windows (%d): %d codes, %d off by one, %d off by more" % (wins, codes, off1, more))
    return 0 if (worst_mel < 1e-3 and more == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
