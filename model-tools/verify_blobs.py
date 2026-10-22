"""Compare every downloaded file's git blob SHA-1 with the original (gated) repo's listing."""
import hashlib
import os
import sys

import truststore

truststore.inject_into_ssl()
from huggingface_hub import HfApi  # noqa: E402

local_dir, original = sys.argv[1], sys.argv[2]
ref = {s.rfilename: s for s in HfApi().model_info(original, files_metadata=True).siblings}
bad = 0
for name in sorted(os.listdir(local_dir)):
    path = os.path.join(local_dir, name)
    if not os.path.isfile(path) or name not in ref:
        print("  (not in original) " + name)
        continue
    data = open(path, "rb").read()
    s = ref[name]
    if s.lfs:
        ok = hashlib.sha256(data).hexdigest() == s.lfs.sha256
    else:
        ok = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == s.blob_id
    bad += not ok
    print(("  OK   " if ok else "  DIFF ") + name)
sys.exit(1 if bad else 0)
