"""Download the Project 17 source models and check mirrors against the gated originals' checksums."""
import os
import sys

import truststore

truststore.inject_into_ssl()
from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

DL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "downloads")  # model-tools/downloads (gitignored)
api = HfApi()


def lfs_hashes(repo):
    info = api.model_info(repo, files_metadata=True)
    return {s.rfilename: (s.lfs.sha256 if s.lfs else None, s.size) for s in info.siblings}


def fetch(repo, original=None, skip=()):
    files = lfs_hashes(repo)
    if original:
        ref = lfs_hashes(original)
        for name, (sha, size) in files.items():
            if name in ref and sha and ref[name][0] and sha != ref[name][0]:
                sys.exit("MISMATCH %s/%s vs %s" % (repo, name, original))
        common = [n for n in files if n in ref and files[n][0]]
        print("%s: %d LFS files match %s" % (repo, len(common), original))
    out = os.path.join(DL, repo.replace("/", "__"))
    for name in files:
        if any(name.endswith(s) or name.startswith(s) for s in skip):
            continue
        hf_hub_download(repo, name, local_dir=out)
    print("downloaded", repo, "->", out)


if __name__ == "__main__":
    fetch("naklitechie/indictrans2-en-indic-dist-200M", original="ai4bharat/indictrans2-en-indic-dist-200M")
    fetch("facebook/mms-tts-hin")
