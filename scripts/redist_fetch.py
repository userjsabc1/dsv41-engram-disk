"""List / download NVIDIA CUDA redistributable components into a private CUDA_HOME.

  python redist_fetch.py list
  python redist_fetch.py get <dest> comp1 comp2 ...
"""
import json
import os
import sys
import tarfile
import urllib.request

BASE = "https://developer.download.nvidia.com/compute/cuda/redist/"
MANIFEST = "redistrib_13.2.2.json"
d = json.load(urllib.request.urlopen(BASE + MANIFEST, timeout=60))


def pick(entry):
    """linux-x86_64 entry; CUDA 13 manifests may nest per-toolkit variants."""
    p = entry.get("linux-x86_64")
    if p and "relative_path" not in p:
        p = p.get("cuda13") or next(iter(p.values()))
    return p

if sys.argv[1] == "list":
    print("release:", d.get("release_date"))
    for k in sorted(d):
        e = d[k]
        if isinstance(e, dict) and "version" in e:
            p = pick(e)
            size = round(int(p["size"]) / 2**20, 1) if p else None
            print(f"{k:28s} {e['version']:12s} {size} MiB")
else:
    dest = os.path.expanduser(sys.argv[2])
    os.makedirs(dest, exist_ok=True)
    for comp in sys.argv[3:]:
        p = pick(d[comp])
        url = BASE + p["relative_path"]
        tmp = f"/tmp/{os.path.basename(url)}"
        print("downloading", comp, d[comp]["version"], round(int(p["size"]) / 2**20, 1), "MiB", flush=True)
        urllib.request.urlretrieve(url, tmp)
        with tarfile.open(tmp) as t:
            top = t.getnames()[0].split("/")[0]
            for m in t.getmembers():
                m.name = m.name[len(top) + 1 :] if m.name.startswith(top + "/") else m.name
                if m.name:
                    t.extract(m, dest, filter="data")
        os.remove(tmp)
    print("done ->", dest)
