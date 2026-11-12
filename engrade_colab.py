# %% [markdown]
# # EnGrade-CP: leakage-audited ordinal grading of enhancer activity with genomic-language-model views and tier-conditional conformal sets
#
# Runs top to bottom on a free Google Colab runtime (T4 GPU recommended; CPU-only also works, just slower).
# Runtime -> Change runtime type -> T4 GPU, then Runtime -> Run all.
#
# What the notebook does
# 1. Downloads the iEnhancer-2L / iEnhancer-EL benchmark exactly as released with Enhancer-FRL, plus an external
#    human enhancer set (Genomic Benchmarks, human_enhancers_cohn) from the Hugging Face Hub.
# 2. Audits the benchmark for leakage (exact duplicates between training and independent test, abutting genomic tiles).
# 3. Re-implements two reference pipelines (stacked multiview features in the spirit of Enhancer-FRL; a CNN-deep-LSTM
#    cascade in the spirit of the hybrid CNN-DLSTM paper) and a frozen genomic-language-model probe.
# 4. Trains EnGrade, a single rank-consistent ordinal model (non-enhancer < weak < strong) that fuses a
#    reverse-complement-invariant convolutional view, a composition view and a frozen DNABERT view through a learned gate.
# 5. Wraps every model with split conformal prediction (marginal and tier-conditional / Mondrian, contiguous ordinal sets).
# 6. Compares a random-split protocol with a chromosome-blocked protocol, runs ablations, an external shift test,
#    gradient-times-input attribution, and writes all tables (CSV) and figures (PNG) to ./engrade_outputs.

# %%
# ---------------------------------------------------------------- 0. Setup
import os, sys, re, time, json, math, random, shutil, subprocess, urllib.request, warnings, itertools
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

def _pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=False)

for mod, pkg in [("transformers", "transformers"), ("huggingface_hub", "huggingface_hub"),
                 ("sklearn", "scikit-learn"), ("pyarrow", "pyarrow")]:
    try:
        __import__(mod)
    except ImportError:
        _pip(pkg)

import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, matthews_corrcoef, f1_score, cohen_kappa_score,
                             accuracy_score, roc_curve)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch", torch.__version__, "| device:", DEVICE,
      "|", torch.cuda.get_device_name(0) if DEVICE.type == "cuda" else "CPU only")
T0 = time.time()

# %%
# ---------------------------------------------------------------- 1. Configuration
CFG = dict(
    out_dir="engrade_outputs",
    data_dir="engrade_data",
    n_folds=5,                 # outer folds for both protocols
    cal_frac=0.20,             # share of each training fold held out for conformal calibration
    alpha=0.10,                # target miscoverage (90% sets)
    seeds_main=[0, 1, 2],      # EnGrade seeds (blocked protocol)
    seeds_other=[0],           # seeds for ablations, CNN-DLSTM and the random-split protocol
    epochs=60, patience=8, batch=64, lr=1e-3, wd=1e-4,
    use_glm=True,              # frozen DNABERT-6 view (falls back to off automatically if download fails)
    glm_name="zhihan1996/DNA_bert_6",
    ext_n=1000,                # external sequences scored (balanced); raise to 6948 on GPU for the full test split
    n_boot=1000,
    quick=False,               # True = 2 folds, 1 seed, 15 epochs smoke test
)
if CFG["quick"]:
    CFG.update(n_folds=2, seeds_main=[0], seeds_other=[0], epochs=15, ext_n=200, n_boot=200)
OUT, DATA = CFG["out_dir"], CFG["data_dir"]
os.makedirs(OUT, exist_ok=True); os.makedirs(DATA, exist_ok=True)
json.dump(CFG, open(f"{OUT}/config.json", "w"), indent=2)

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# %%
# ---------------------------------------------------------------- 2. Data download
RAR_URL = "https://raw.githubusercontent.com/wangchao-malab/Enhancer-FRL/main/training_testing_datasets.rar"
FASTAS = ["enhancers_nonenhancers_training.fasta", "enhancers_nonenhancers_testing.fasta",
          "strong_weak_training.fasta", "strong_weak_testing.fasta"]

def _extract_rar(rar, dst):
    # 1) system unrar (present on Colab)  2) unrar-cffi wheel  3) apt-get unrar
    if shutil.which("unrar"):
        r = subprocess.run(["unrar", "e", "-o+", "-inul", rar, dst + "/"], check=False)
        if all(os.path.exists(f"{dst}/{f}") for f in FASTAS): return
    try:
        from unrar.cffi import rarfile as cr
    except ImportError:
        _pip("unrar-cffi")
        try:
            from unrar.cffi import rarfile as cr
        except ImportError:
            cr = None
    if cr is not None:
        rf = cr.RarFile(rar)
        for n in rf.namelist():
            b = os.path.basename(n)
            if b in FASTAS:
                open(f"{dst}/{b}", "wb").write(rf.read(n))
        if all(os.path.exists(f"{dst}/{f}") for f in FASTAS): return
    subprocess.run("apt-get -qq install -y unrar > /dev/null 2>&1", shell=True, check=False)
    subprocess.run(["unrar", "e", "-o+", "-inul", rar, dst + "/"], check=False)
    if not all(os.path.exists(f"{dst}/{f}") for f in FASTAS):
        raise RuntimeError("Could not extract the benchmark archive")

if not all(os.path.exists(f"{DATA}/{f}") for f in FASTAS):
    rar = f"{DATA}/training_testing_datasets.rar"
    urllib.request.urlretrieve(RAR_URL, rar)
    _extract_rar(rar, DATA)

def read_fasta(p):
    out, h = [], None
    for line in open(p):
        line = line.strip().replace("\r", "")
        if not line: continue
        if line.startswith(">"): h = line[1:]
        else: out.append((h, line.upper()))
    return out

tr_en = read_fasta(f"{DATA}/enhancers_nonenhancers_training.fasta")
te_en = read_fasta(f"{DATA}/enhancers_nonenhancers_testing.fasta")
tr_sw = read_fasta(f"{DATA}/strong_weak_training.fasta")
te_sw = read_fasta(f"{DATA}/strong_weak_testing.fasta")

strong_map = {s: int(h.split("|")[1]) for h, s in tr_sw}   # 1 = strong, 0 = weak
def chrom_of(h):
    m = re.match(r"(chr[0-9xy]+)_(\d+)_(\d+)", h.split("|")[0], re.I)
    return (m.group(1).lower(), int(m.group(2)), int(m.group(3))) if m else (None, None, None)

rows = []
for h, s in tr_en:
    lab = int(h.split("|")[1])
    tier = 0 if lab == 0 else (2 if strong_map[s] == 1 else 1)
    c, a, b = chrom_of(h)
    rows.append(dict(name=h.split("|")[0], seq=s, tier=tier, chrom=c, start=a, end=b))
train = pd.DataFrame(rows)
assert (train.tier > 0).sum() == len(strong_map), "strength labels must cover every training enhancer"

# Independent set (iEnhancer-EL convention): 100 strong, 100 weak, 200 non-enhancers, in this order.
assert [s for _, s in te_sw] == [s for _, s in te_en][:200], "independent-set order differs from the expected convention"
ind = pd.DataFrame(dict(name=[h.split("|")[0] for h, _ in te_en], seq=[s for _, s in te_en],
                        tier=[2] * 100 + [1] * 100 + [0] * 200))
train_set = set(train.seq)
ind["dup_in_train"] = ind.seq.isin(train_set)
print("training:", train.tier.value_counts().sort_index().to_dict(),
      "| independent:", ind.tier.value_counts().sort_index().to_dict())

# External: Genomic Benchmarks human_enhancers_cohn (500 bp, label 1 = enhancer)
EXT_OK = True
try:
    from huggingface_hub import list_repo_files, hf_hub_download
    repo = "katarinagresova/Genomic_Benchmarks_human_enhancers_cohn"
    f_test = [f for f in list_repo_files(repo, repo_type="dataset") if f.startswith("data/test")][0]
    ext_all = pd.read_parquet(hf_hub_download(repo, f_test, repo_type="dataset"))
    ext_all["seq"] = ext_all["seq"].str.upper()
    per = CFG["ext_n"] // 2
    ext = pd.concat([ext_all[ext_all.label == k].sample(min(per, (ext_all.label == k).sum()), random_state=0)
                     for k in (0, 1)]).reset_index(drop=True)
    print("external (cohn) scored:", ext.label.value_counts().to_dict())
except Exception as e:
    EXT_OK = False; ext = None
    print("External dataset unavailable, skipping the shift test:", repr(e))

# %%
# ---------------------------------------------------------------- 3. Leakage audit
def kset(s, k=8): return {s[i:i + k] for i in range(len(s) - k + 1)}
tr_k = [kset(s) for s in train.seq]
def max_containment(s):
    q = kset(s); return max(len(q & t) / max(1, len(q)) for t in tr_k)
ind["max_8mer_containment"] = [max_containment(s) for s in ind.seq]

loc = train.dropna(subset=["chrom"]).sort_values(["chrom", "start"])
abut = 0; overlap = 0
for _, g in loc.groupby("chrom"):
    st, en = g.start.values, g.end.values
    abut += int(np.sum(st[1:] == en[:-1])); overlap += int(np.sum(st[1:] < en[:-1]))
# loci = runs of abutting/overlapping tiles
locus_id, lid = {}, 0
for _, g in loc.groupby("chrom"):
    prev_end = None
    for i, r in g.iterrows():
        if prev_end is None or r.start > prev_end: lid += 1
        locus_id[i] = lid; prev_end = max(prev_end or 0, r.end)
train["locus"] = pd.Series(locus_id)
loc_sizes = train.groupby("locus").size()
multi_tile_loci = int((loc_sizes > 1).sum()); tiles_in_multi = int(loc_sizes[loc_sizes > 1].sum())

# within-locus label purity
purity = train.groupby("locus").tier.agg(lambda t: t.value_counts(normalize=True).iloc[0])[loc_sizes > 1].mean()

audit = pd.DataFrame([
    ("independent sequences that are exact copies of training sequences", int(ind.dup_in_train.sum())),
    ("  of which strong", int(ind[ind.tier == 2].dup_in_train.sum())),
    ("  of which weak", int(ind[ind.tier == 1].dup_in_train.sum())),
    ("  of which non-enhancer", int(ind[ind.tier == 0].dup_in_train.sum())),
    ("independent sequences with >50% 8-mer containment in a training sequence", int((ind.max_8mer_containment > 0.5).sum())),
    ("training tiles abutting the previous tile on the same chromosome", abut),
    ("training tiles overlapping the previous tile", overlap),
    ("loci made of >1 abutting tile", multi_tile_loci),
    ("training tiles inside multi-tile loci", tiles_in_multi),
    ("mean tier purity inside multi-tile loci", round(float(purity), 3)),
    ("distinct chromosomes in training", int(train.chrom.nunique())),
], columns=["check", "value"])
audit.to_csv(f"{OUT}/T1_leakage_audit.csv", index=False)
print(audit.to_string(index=False))
ind["clean"] = ~ind.dup_in_train
print("clean independent subset:", ind[ind.clean].tier.value_counts().sort_index().to_dict())

# %%
# ---------------------------------------------------------------- 4. Sequence encodings
L = 200
LUT = np.full(256, -1, dtype=np.int64)
for i, ch in enumerate("ACGT"): LUT[ord(ch)] = i
NCP = np.array([[1, 1, 1], [0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
EIIP = np.array([0.1260, 0.1340, 0.0806, 0.1335], dtype=np.float32)
COMP = str.maketrans("ACGTN", "TGCAN")

def fix_len(s): return (s + "N" * L)[:L]
def rc(s): return s.translate(COMP)[::-1]
def to_int(s): return LUT[np.frombuffer(s.encode(), dtype=np.uint8)]

def channels(s):
    a = to_int(s); v = a >= 0; x = np.zeros((8, len(a)), np.float32)
    x[a[v], np.where(v)[0]] = 1.0
    x[4:7, v] = NCP[a[v]].T; x[7, v] = EIIP[a[v]]
    return x

def kmer_freq(a, k):
    n = len(a) - k + 1
    W = np.stack([a[i:i + n] for i in range(k)], 1)
    ok = (W >= 0).all(1); idx = (W[ok] * (4 ** np.arange(k - 1, -1, -1))).sum(1)
    return np.bincount(idx, minlength=4 ** k) / max(1, ok.sum())

def cksnap(a, gmax=5):
    out = []
    for g in range(gmax + 1):
        x, y = a[:-(g + 1)], a[g + 1:]; ok = (x >= 0) & (y >= 0)
        out.append(np.bincount(x[ok] * 4 + y[ok], minlength=16) / max(1, ok.sum()))
    return np.concatenate(out)

def enac(a, w=5):
    oh = np.zeros((len(a), 4), np.float32); v = a >= 0; oh[np.where(v)[0], a[v]] = 1
    cs = np.vstack([np.zeros((1, 4)), np.cumsum(oh, 0)])
    return ((cs[w:] - cs[:-w]) / w).ravel()

def ncp_anf(a):
    out = np.zeros((len(a), 4), np.float32); cnt = np.zeros(4)
    for i, b in enumerate(a):
        if b >= 0:
            cnt[b] += 1; out[i, :3] = NCP[b]; out[i, 3] = cnt[b] / (i + 1)
    return out.ravel()

TRI_EIIP = np.array([EIIP[i] + EIIP[j] + EIIP[k] for i in range(4) for j in range(4) for k in range(4)])
def pse_eiip(a): return kmer_freq(a, 3) * TRI_EIIP

def prop_autocov(a, lags=10):
    v = a >= 0; P = np.zeros((len(a), 4), np.float32); P[v, :3] = NCP[a[v]]; P[v, 3] = EIIP[a[v]]
    P = P - P.mean(0); out = []
    for lag in range(1, lags + 1): out.append((P[:-lag] * P[lag:]).mean(0))
    return np.concatenate(out)

VIEW_FUNCS = {
    "Kmer": lambda a: np.concatenate([kmer_freq(a, k) for k in (1, 2, 3, 4)]),
    "CKSNAP": cksnap, "ENAC": enac, "NCP-ANF": ncp_anf, "PseEIIP": pse_eiip, "PAC": prop_autocov,
}
COMP_VIEWS = ["Kmer", "CKSNAP", "PseEIIP", "PAC"]   # position-free composition view used inside EnGrade

def encode(seqs):
    seqs = [fix_len(s) for s in seqs]
    ints = [to_int(s) for s in seqs]
    views = {k: np.array([f(a) for a in ints], dtype=np.float32) for k, f in VIEW_FUNCS.items()}
    X = np.stack([channels(s) for s in seqs]); Xrc = np.stack([channels(rc(s)) for s in seqs])
    K3 = np.array([[(a[i] * 16 + a[i + 1] * 4 + a[i + 2]) if min(a[i:i + 3]) >= 0 else 64
                    for i in range(L - 2)] for a in ints], dtype=np.int64)
    return dict(views=views, X=X, Xrc=Xrc, K3=K3, seqs=seqs)

t = time.time()
ENC_TR = encode(train.seq.tolist()); ENC_IND = encode(ind.seq.tolist())
EXT_WIN = [0, 150, 300]
if EXT_OK:
    ext_windows = [s[o:o + L] for s in ext.seq for o in EXT_WIN]
    ENC_EXT = encode(ext_windows)
print("encodings ready in %.1fs; view dims:" % (time.time() - t),
      {k: v.shape[1] for k, v in ENC_TR["views"].items()})

# %%
# ---------------------------------------------------------------- 5. Frozen genomic language model view (DNABERT-6)
def glm_embed(seqs, tag):
    path = f"{DATA}/glm_{tag}_{len(seqs)}.npy"
    if os.path.exists(path): return np.load(path)
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained(CFG["glm_name"])
    mdl = AutoModel.from_pretrained(CFG["glm_name"]).to(DEVICE).eval()
    bs = 64 if DEVICE.type == "cuda" else 16; out = []
    for i in range(0, len(seqs), bs):
        batch = [" ".join(s[j:j + 6] for j in range(len(s) - 5)) for s in seqs[i:i + bs]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad(), torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=DEVICE.type == "cuda"):
            hs = mdl(**enc).last_hidden_state.float()
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((hs * m).sum(1) / m.sum(1)).cpu().numpy())
    E = np.concatenate(out).astype(np.float32); np.save(path, E)
    del mdl; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None
    return E

GLM_OK = False
if CFG["use_glm"]:
    try:
        t = time.time()
        G_TR = glm_embed(ENC_TR["seqs"], "train"); G_IND = glm_embed(ENC_IND["seqs"], "ind")
        G_EXT = glm_embed(ENC_EXT["seqs"], "ext") if EXT_OK else None
        GLM_OK = True; print("gLM embeddings:", G_TR.shape, "in %.1fs" % (time.time() - t))
    except Exception as e:
        print("gLM view unavailable, continuing without it:", repr(e))
if not GLM_OK:
    G_TR = np.zeros((len(train), 8), np.float32); G_IND = np.zeros((len(ind), 8), np.float32)
    G_EXT = np.zeros((len(ENC_EXT["seqs"]), 8), np.float32) if EXT_OK else None
ENC_TR["G"], ENC_IND["G"] = G_TR, G_IND
if EXT_OK: ENC_EXT["G"] = G_EXT
for E in [ENC_TR, ENC_IND] + ([ENC_EXT] if EXT_OK else []):
    E["H"] = np.concatenate([E["views"][k] for k in COMP_VIEWS], 1)

# %%
# ---------------------------------------------------------------- 6. Models
class RCConv(nn.Module):
    def __init__(self, cin=8, d=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, 96, 9, padding=4), nn.BatchNorm1d(96), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(96, 96, 7, padding=3), nn.BatchNorm1d(96), nn.GELU(), nn.MaxPool1d(2),
            nn.Conv1d(96, 128, 5, padding=4, dilation=2), nn.BatchNorm1d(128), nn.GELU())
        self.proj = nn.Linear(256, d)
    def embed(self, x):
        h = self.net(x); return self.proj(torch.cat([h.amax(-1), h.mean(-1)], 1))

class EnGrade(nn.Module):
    """Ordinal grader over gated views. views: subset of {'cnn','comp','glm'}.
    head: 'corn' (conditional ordinal, default), 'coral' (shared slope), 'softmax' (nominal)."""
    def __init__(self, d_comp, d_glm, views=("cnn", "comp", "glm"), gate=True, rc=True, head="corn",
                 d=128, view_drop=0.15):
        super().__init__()
        self.views, self.gate_on, self.rc, self.head, self.view_drop = list(views), gate, rc, head, view_drop
        if "cnn" in views: self.cnn = RCConv(d=d)
        if "comp" in views: self.comp = nn.Sequential(nn.Linear(d_comp, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, d))
        if "glm" in views: self.glm = nn.Sequential(nn.LayerNorm(d_glm), nn.Linear(d_glm, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, d))
        V = len(self.views)
        self.gate = nn.Linear(d * V, V) if gate else None
        self.mix = None if gate else nn.Linear(d * V, d)
        self.drop = nn.Dropout(0.2)
        if head == "coral":
            self.w = nn.Linear(d, 1, bias=False); self.b = nn.Parameter(torch.tensor([0.5, -0.5]))
        else:
            self.out = nn.Linear(d, 2 if head == "corn" else 3)
        self.last_gate = None
    def forward(self, B):
        Z = []
        for v in self.views:
            if v == "cnn":
                z = self.cnn.embed(B["X"])
                if self.rc: z = 0.5 * (z + self.cnn.embed(B["Xrc"]))
            elif v == "comp": z = self.comp(B["H"])
            else: z = self.glm(B["G"])
            Z.append(z)
        Zs = torch.stack(Z, 1)                                   # (n, V, d)
        if self.gate is not None:
            logits = self.gate(Zs.flatten(1))
            if self.training and len(self.views) > 1 and self.view_drop > 0:
                mask = (torch.rand_like(logits) < self.view_drop)
                mask[mask.all(1)] = False
                logits = logits.masked_fill(mask, -1e4)
            g = torch.softmax(logits, 1); self.last_gate = g.detach()
            z = (g.unsqueeze(-1) * Zs).sum(1)
        else:
            z = self.mix(Zs.flatten(1))
        z = self.drop(F.gelu(z))
        return self.w(z) + self.b if self.head == "coral" else self.out(z)
    @staticmethod
    def to_probs(o, head):
        if head == "coral":
            c = torch.sigmoid(o)                                 # P(y>0), P(y>1)
            c2 = torch.minimum(c[:, 1], c[:, 0])
            return torch.stack([1 - c[:, 0], c[:, 0] - c2, c2], 1)
        if head == "corn":
            p1, q = torch.sigmoid(o[:, 0]), torch.sigmoid(o[:, 1])   # P(y>0), P(y>1 | y>0)
            return torch.stack([1 - p1, p1 * (1 - q), p1 * q], 1)
        return torch.softmax(o, 1)

class CNNDLSTM(nn.Module):
    """Binary CNN + stacked LSTM over one-hot plus learned 3-mer embeddings (reference re-implementation)."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(65, 100, padding_idx=64)
        self.c1 = nn.Conv1d(104, 16, 11, padding=5); self.c2 = nn.Conv1d(16, 8, 7, padding=3)
        self.lstm = nn.LSTM(8, 64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(64, 1))
    def forward(self, B):
        x = torch.cat([B["X"][:, :4, :L - 2], self.emb(B["K3"]).transpose(1, 2)], 1)
        h = F.max_pool1d(F.relu(self.c2(F.relu(self.c1(x)))), 2).transpose(1, 2)
        o, _ = self.lstm(h); return self.fc(o[:, -1]).squeeze(-1)

def batch_of(E, idx, sc_h=None, sc_g=None):
    H = E["H"][idx]; G = E["G"][idx]
    if sc_h is not None: H = sc_h.transform(H)
    if sc_g is not None: G = sc_g.transform(G)
    return {"X": torch.tensor(E["X"][idx], device=DEVICE), "Xrc": torch.tensor(E["Xrc"][idx], device=DEVICE),
            "K3": torch.tensor(E["K3"][idx], device=DEVICE),
            "H": torch.tensor(H, dtype=torch.float32, device=DEVICE), "G": torch.tensor(G, dtype=torch.float32, device=DEVICE)}

def train_torch(make, E, y, fit_idx, kind, seed):
    """kind: 'corn' | 'coral' | 'softmax' | 'binary'. Returns (model, scalers)."""
    set_seed(seed)
    tr_i, va_i = train_test_split(fit_idx, test_size=0.1, random_state=seed, stratify=y[fit_idx])
    sc_h = StandardScaler().fit(E["H"][tr_i]); sc_g = StandardScaler().fit(E["G"][tr_i])
    m = make().to(DEVICE); opt = torch.optim.AdamW(m.parameters(), lr=CFG["lr"], weight_decay=CFG["wd"])
    Btr, Bva = batch_of(E, tr_i, sc_h, sc_g), batch_of(E, va_i, sc_h, sc_g)
    ytr, yva = torch.tensor(y[tr_i], device=DEVICE), torch.tensor(y[va_i], device=DEVICE)
    if kind == "corn":
        def lossf(o, t):
            l1 = F.binary_cross_entropy_with_logits(o[:, 0], (t > 0).float())
            e = t > 0
            l2 = F.binary_cross_entropy_with_logits(o[e, 1], (t[e] > 1).float()) if e.any() else o.sum() * 0
            return l1 + l2
    elif kind == "coral":
        lossf = lambda o, t: F.binary_cross_entropy_with_logits(o, torch.stack([(t > 0), (t > 1)], 1).float())
    elif kind == "softmax":
        cw = torch.tensor([len(ytr) / (3 * max(1, (ytr == k).sum().item())) for k in range(3)], device=DEVICE)
        lossf = lambda o, t: F.cross_entropy(o, t, weight=cw)
    else:
        pw = torch.tensor((ytr == 0).sum().item() / max(1, (ytr == 1).sum().item()), device=DEVICE)
        lossf = lambda o, t: F.binary_cross_entropy_with_logits(o, t.float(), pos_weight=pw)
    best, best_state, bad, n = 1e9, None, 0, len(tr_i)
    for ep in range(CFG["epochs"]):
        m.train(); perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, CFG["batch"]):
            j = perm[i:i + CFG["batch"]]
            if len(j) < 2: continue
            loss = lossf(m({k: v[j] for k, v in Btr.items()}), ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = lossf(m(Bva), yva).item()
        if vl < best - 1e-4: best, bad, best_state = vl, 0, {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]: break
    m.load_state_dict(best_state); m.eval()
    return m, (sc_h, sc_g)

@torch.no_grad()
def predict_torch(m, sc, E, idx, kind):
    out = []
    for i in range(0, len(idx), 512):
        o = m(batch_of(E, idx[i:i + 512], *sc))
        out.append((EnGrade.to_probs(o, kind) if kind != "binary" else torch.sigmoid(o)).cpu().numpy())
    return np.concatenate(out)

def cascade_probs(p1, p2):
    """p1 = P(enhancer), p2 = P(strong | enhancer) -> 3-tier probabilities."""
    return np.stack([1 - p1, p1 * (1 - p2), p1 * p2], 1)

# ---- stacked multiview baseline (Enhancer-FRL style, 6 encodings x 4 learners -> logistic meta-learner)
BASE_LEARNERS = {
    "SVM": lambda: make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale")),
    "RF": lambda: RandomForestClassifier(n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=0),
    "HGB": lambda: HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, random_state=0),
    "KNN": lambda: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
}
def _score(m, X): return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]

class StackFRL:
    def fit(self, views, y):
        names = list(views); n = len(y); oof = np.zeros((n, len(names) * len(BASE_LEARNERS)))
        skf = StratifiedKFold(3, shuffle=True, random_state=0); self.models = []
        col = 0
        for v in names:
            for ln, mk in BASE_LEARNERS.items():
                for a, b in skf.split(views[v], y):
                    oof[b, col] = _score(mk().fit(views[v][a], y[a]), views[v][b])
                self.models.append((v, mk().fit(views[v], y))); col += 1
        self.sc = StandardScaler().fit(oof)
        self.meta = LogisticRegression(C=1.0, max_iter=2000).fit(self.sc.transform(oof), y)
        return self
    def predict(self, views):
        S = np.column_stack([_score(m, views[v]) for v, m in self.models])
        return self.meta.predict_proba(self.sc.transform(S))[:, 1]

def subviews(E, idx): return {k: v[idx] for k, v in E["views"].items()}

# %%
# ---------------------------------------------------------------- 7. Conformal prediction (ordinal, contiguous sets)
def conformal_sets(P_cal, y_cal, P_te, alpha, mondrian=True):
    S = np.zeros_like(P_te, dtype=bool)
    if mondrian:
        for k in range(3):
            s = 1 - P_cal[y_cal == k, k]; n = len(s)
            q = np.inf if n == 0 else np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher")
            S[:, k] = (1 - P_te[:, k]) <= q
    else:
        s = 1 - P_cal[np.arange(len(y_cal)), y_cal]; n = len(s)
        q = np.quantile(s, min(1.0, math.ceil((n + 1) * (1 - alpha)) / n), method="higher")
        S = (1 - P_te) <= q
    empty = ~S.any(1); S[empty, P_te[empty].argmax(1)] = True
    lo = S.argmax(1); hi = 2 - S[:, ::-1].argmax(1)                # make sets contiguous on the ordinal scale
    for i in range(len(S)): S[i, lo[i]:hi[i] + 1] = True
    return S

def set_stats(S, y):
    cov = S[np.arange(len(y)), y]; size = S.sum(1)
    d = dict(coverage=cov.mean(), mean_size=size.mean(), singleton=(size == 1).mean(),
             singleton_acc=(cov[size == 1].mean() if (size == 1).any() else np.nan))
    for k, nm in enumerate(["non", "weak", "strong"]):
        if (y == k).any(): d[f"cov_{nm}"] = cov[y == k].mean(); d[f"size_{nm}"] = size[y == k].mean()
    return d

# %%
# ---------------------------------------------------------------- 8. Metrics
def mcc_at(y, s, thr=0.5): return matthews_corrcoef(y, (s >= thr).astype(int))
def bin_block(y, s):
    yh = (s >= 0.5).astype(int); tp = ((yh == 1) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
    return dict(ACC=accuracy_score(y, yh), SEN=tp / max(1, (y == 1).sum()), SPE=tn / max(1, (y == 0).sum()),
                MCC=matthews_corrcoef(y, yh), AUC=roc_auc_score(y, s) if len(set(y)) > 1 else np.nan)

def tier_metrics(P, y):
    d = {}
    yh = P.argmax(1)
    d["macroF1"] = f1_score(y, yh, average="macro"); d["QWK"] = cohen_kappa_score(y, yh, weights="quadratic")
    d["tierACC"] = accuracy_score(y, yh)
    for k, v in bin_block((y > 0).astype(int), P[:, 1] + P[:, 2]).items(): d["L1_" + k] = v
    e = y > 0
    if e.sum() > 0 and len(set(y[e])) > 1:
        s2 = P[e, 2] / np.clip(P[e, 1] + P[e, 2], 1e-9, None)
        for k, v in bin_block((y[e] == 2).astype(int), s2).items(): d["L2_" + k] = v
    return d

def boot_ci(P, y, key, n=None, seed=0):
    n = n or CFG["n_boot"]; rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        try: vals.append(tier_metrics(P[i], y[i]).get(key, np.nan))
        except Exception: pass
    return np.nanpercentile(vals, [2.5, 97.5])

def paired_boot(PA, PB, y, key, n=None, seed=1):
    n = n or CFG["n_boot"]; rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        d.append(tier_metrics(PA[i], y[i])[key] - tier_metrics(PB[i], y[i])[key])
    d = np.array(d); return d.mean(), np.percentile(d, [2.5, 97.5]), 2 * min((d <= 0).mean(), (d >= 0).mean())

# %%
# ---------------------------------------------------------------- 9. Experiment runner
y_all = train.tier.values.astype(np.int64)
N = len(y_all); IDX_IND = np.arange(len(ind))
IDX_EXT = np.arange(len(ENC_EXT["seqs"])) if EXT_OK else None
d_comp, d_glm = ENC_TR["H"].shape[1], ENC_TR["G"].shape[1]
FULL_VIEWS = ("cnn", "comp", "glm") if GLM_OK else ("cnn", "comp")

ENGRADE_VARIANTS = {
    "EnGrade": dict(views=FULL_VIEWS),
    "w/o gLM view": dict(views=("cnn", "comp")),
    "w/o composition view": dict(views=tuple(v for v in FULL_VIEWS if v != "comp")),
    "w/o convolutional view": dict(views=tuple(v for v in FULL_VIEWS if v != "cnn")),
    "w/o gate (concat)": dict(views=FULL_VIEWS, gate=False),
    "w/o RC averaging": dict(views=FULL_VIEWS, rc=False),
    "shared-slope ordinal head (CORAL)": dict(views=FULL_VIEWS, head="coral"),
    "nominal softmax head": dict(views=FULL_VIEWS, head="softmax"),
}
if not GLM_OK: ENGRADE_VARIANTS.pop("w/o gLM view")

def make_folds(kind):
    if kind == "random":
        return list(StratifiedKFold(CFG["n_folds"], shuffle=True, random_state=0).split(np.zeros(N), y_all))
    return list(StratifiedGroupKFold(CFG["n_folds"], shuffle=True, random_state=0).split(np.zeros(N), y_all, groups=train.chrom.values))

# Robustness probe: a fixed stratified subset of training sequences is re-scored by each held-out-fold model
# after random point substitutions (rates below). Perturbation uses its own RNG, so training is unaffected.
ROB_RATES = [0.02, 0.05, 0.10]
_, ROB_IDX = train_test_split(np.arange(N), test_size=min(600, N // 4), random_state=7, stratify=y_all)
ROB_IDX = np.sort(ROB_IDX); ROB_POS = {int(i): j for j, i in enumerate(ROB_IDX)}
def mutate(seq, rate, rng):
    s = np.array(list(seq)); m = rng.random(len(s)) < rate
    for i in np.where(m)[0]: s[i] = rng.choice([b for b in "ACGT" if b != s[i]])
    return "".join(s)
ENC_ROB = {}
for r in ROB_RATES:
    rng = np.random.default_rng(int(r * 1000))
    seqs = [mutate(ENC_TR["seqs"][i], r, rng) for i in ROB_IDX]
    E = encode(seqs)
    E["G"] = glm_embed(E["seqs"], f"rob{int(r * 100)}") if GLM_OK else np.zeros((len(seqs), ENC_TR["G"].shape[1]), np.float32)
    E["H"] = np.concatenate([E["views"][k] for k in COMP_VIEWS], 1); ENC_ROB[r] = E
ROB_RES = {}  # (model, seed, rate) -> (len(ROB_IDX), 3)
def rob_store(name, seed, pr, te):
    sel = np.array([ROB_POS[int(i)] for i in te if int(i) in ROB_POS], dtype=int)
    if len(sel) == 0: return
    clean = ROB_RES.setdefault((name, seed, 0.0), np.full((len(ROB_IDX), 3), np.nan))
    clean[sel] = pr(ENC_TR, ROB_IDX[sel])
    for r in ROB_RATES:
        a = ROB_RES.setdefault((name, seed, r), np.full((len(ROB_IDX), 3), np.nan)); a[sel] = pr(ENC_ROB[r], sel)
FIT_TIME = {}  # (model) -> list of per-fold training seconds (blocked protocol)
def tick(name, t0): FIT_TIME.setdefault(name, []).append(time.time() - t0)

RES = {}      # (protocol, model, seed) -> dict(oof, ind[list], ext[list], cal[list])
GATES = []
def store(key, fold, te, P_te, P_cal, y_cal, P_ind, P_ext):
    r = RES.setdefault(key, dict(oof=np.full((N, 3), np.nan), ind=[], ext=[], cal=[]))
    r["oof"][te] = P_te; r["ind"].append(P_ind); r["cal"].append((fold, P_cal, y_cal, te))
    if P_ext is not None: r["ext"].append(P_ext)

def agg_ext(P):  # (n_seq*3 windows, 3) -> pick the window with max enhancer probability
    W = P.reshape(-1, len(EXT_WIN), 3); j = (W[:, :, 1] + W[:, :, 2]).argmax(1)
    return W[np.arange(len(W)), j]

def run_protocol(proto, models, variants, seeds_engrade):
    folds = make_folds(proto)
    for f, (tr, te) in enumerate(folds):
        fit, cal = train_test_split(tr, test_size=CFG["cal_frac"], random_state=100 + f, stratify=y_all[tr])
        t = time.time()
        # stacked multiview cascade
        if "StackFRL" in models:
            t0 = time.time()
            yb = (y_all[fit] > 0).astype(int); e_fit = fit[y_all[fit] > 0]
            m1 = StackFRL().fit(subviews(ENC_TR, fit), yb)
            m2 = StackFRL().fit(subviews(ENC_TR, e_fit), (y_all[e_fit] == 2).astype(int))
            def pr(E, idx, m1=m1, m2=m2): return cascade_probs(m1.predict(subviews(E, idx)), m2.predict(subviews(E, idx)))
            if proto == "blocked":
                tick("StackFRL (multiview stacking)", t0); rob_store("StackFRL (multiview stacking)", 0, pr, te)
                MODELS_LIVE["StackFRL (multiview stacking)"] = pr
            Pext = agg_ext(pr(ENC_EXT, IDX_EXT)) if EXT_OK else None
            store((proto, "StackFRL (multiview stacking)", 0), f, te, pr(ENC_TR, te), pr(ENC_TR, cal), y_all[cal], pr(ENC_IND, IDX_IND), Pext)
        # gLM linear probe cascade
        if "gLM-probe" in models and GLM_OK:
            t0 = time.time(); yb = (y_all[fit] > 0).astype(int); e_fit = fit[y_all[fit] > 0]
            p1 = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(ENC_TR["G"][fit], yb)
            p2 = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000)).fit(ENC_TR["G"][e_fit], (y_all[e_fit] == 2).astype(int))
            def pr(E, idx, p1=p1, p2=p2): return cascade_probs(p1.predict_proba(E["G"][idx])[:, 1], p2.predict_proba(E["G"][idx])[:, 1])
            if proto == "blocked":
                tick("gLM probe (DNABERT-6 + LR)", t0); rob_store("gLM probe (DNABERT-6 + LR)", 0, pr, te)
                MODELS_LIVE["gLM probe (DNABERT-6 + LR)"] = pr
            Pext = agg_ext(pr(ENC_EXT, IDX_EXT)) if EXT_OK else None
            store((proto, "gLM probe (DNABERT-6 + LR)", 0), f, te, pr(ENC_TR, te), pr(ENC_TR, cal), y_all[cal], pr(ENC_IND, IDX_IND), Pext)
        # CNN-DLSTM cascade
        if "CNN-DLSTM" in models:
            for s in CFG["seeds_other"]:
                t0 = time.time(); yb = (y_all > 0).astype(np.int64)
                m1, sc1 = train_torch(CNNDLSTM, ENC_TR, yb, fit, "binary", s)
                e_fit = fit[y_all[fit] > 0]; ys = (y_all == 2).astype(np.int64)
                m2, sc2 = train_torch(CNNDLSTM, ENC_TR, ys, e_fit, "binary", s)
                def pr(E, idx, m1=m1, m2=m2, sc1=sc1, sc2=sc2): return cascade_probs(predict_torch(m1, sc1, E, idx, "binary"), predict_torch(m2, sc2, E, idx, "binary"))
                if proto == "blocked":
                    tick("CNN-DLSTM cascade", t0); rob_store("CNN-DLSTM cascade", s, pr, te)
                    MODELS_LIVE["CNN-DLSTM cascade"] = pr
                Pext = agg_ext(pr(ENC_EXT, IDX_EXT)) if EXT_OK else None
                store((proto, "CNN-DLSTM cascade", s), f, te, pr(ENC_TR, te), pr(ENC_TR, cal), y_all[cal], pr(ENC_IND, IDX_IND), Pext)
        # EnGrade and its ablations
        for vname in variants:
            kw = ENGRADE_VARIANTS[vname]; head = kw.get("head", "corn")
            for s in (seeds_engrade if vname == "EnGrade" else CFG["seeds_other"]):
                mk = lambda: EnGrade(d_comp, d_glm, **kw)
                t0 = time.time(); m, sc = train_torch(mk, ENC_TR, y_all, fit, head, s)
                pr = lambda E, idx, m=m, sc=sc, head=head: predict_torch(m, sc, E, idx, head)
                if proto == "blocked" and vname == "EnGrade":
                    tick("EnGrade", t0); MODELS_LIVE["EnGrade"] = pr
                    if s == seeds_engrade[0]: rob_store("EnGrade", s, pr, te)
                P_te = pr(ENC_TR, te)
                if vname == "EnGrade" and proto == "blocked" and len(kw["views"]) > 1:
                    GATES.append(pd.DataFrame(_gates(m, sc, te), columns=list(kw["views"])).assign(tier=y_all[te], fold=f, seed=s))
                if vname == "EnGrade" and proto == "blocked" and f == 0 and s == seeds_engrade[0]:
                    ATTR_MODEL.update(model=m, sc=sc, te=te)
                Pext = agg_ext(pr(ENC_EXT, IDX_EXT)) if EXT_OK else None
                store((proto, vname, s), f, te, P_te, pr(ENC_TR, cal), y_all[cal], pr(ENC_IND, IDX_IND), Pext)
        print(f"[{proto}] fold {f + 1}/{len(folds)} done in {time.time() - t:.1f}s  (elapsed {time.time() - T0:.0f}s)")

@torch.no_grad()
def _gates(m, sc, idx):
    m(batch_of(ENC_TR, idx, *sc)); return m.last_gate.cpu().numpy()

ATTR_MODEL = {}; MODELS_LIVE = {}
BASELINES = ["StackFRL", "gLM-probe", "CNN-DLSTM"]
run_protocol("blocked", BASELINES, list(ENGRADE_VARIANTS), CFG["seeds_main"])
LIVE_BLOCKED = dict(MODELS_LIVE)          # last blocked-fold models, used for latency measurement
run_protocol("random", BASELINES, ["EnGrade"], CFG["seeds_other"])
import pickle
pickle.dump(dict(RES=RES, ROB_RES=ROB_RES, ROB_IDX=ROB_IDX, y_all=y_all, y_ind=ind.tier.values, clean=ind.clean.values,
                 y_ext=(ext.label.values if EXT_OK else None), GATES=GATES, FIT_TIME=FIT_TIME, chrom=train.chrom.values,
                 loc_sizes=loc_sizes.values), open(f"{OUT}/predictions.pkl", "wb"))

# %%
# ---------------------------------------------------------------- 10. Results tables
def mean_P(proto, model):
    ks = [k for k in RES if k[0] == proto and k[1] == model]
    return np.mean([RES[k]["oof"] for k in ks], 0), ks

MODELS_MAIN = [m for m in ["StackFRL (multiview stacking)", "CNN-DLSTM cascade", "gLM probe (DNABERT-6 + LR)", "EnGrade"]
               if any(k[1] == m for k in RES)]
rows = []
for proto in ["random", "blocked"]:
    for mname in MODELS_MAIN:
        P, ks = mean_P(proto, mname)
        if not ks: continue
        d = tier_metrics(P, y_all); d.update(protocol=proto, model=mname, n_seeds=len(ks))
        # spread over folds (seed-averaged)
        folds = make_folds(proto)
        fm = [tier_metrics(P[te], y_all[te]) for _, te in folds]
        for key in ["L1_MCC", "L2_MCC", "QWK"]:
            d[key + "_fold_sd"] = np.std([x.get(key, np.nan) for x in fm])
        if proto == "blocked":
            for key in ["L1_MCC", "L2_MCC", "QWK"]:
                lo, hi = boot_ci(P, y_all, key); d[key + "_CI"] = f"[{lo:.3f}, {hi:.3f}]"
        rows.append(d)
T2 = pd.DataFrame(rows)
cols = ["protocol", "model", "n_seeds", "L1_ACC", "L1_SEN", "L1_SPE", "L1_MCC", "L1_AUC", "L2_ACC", "L2_SEN", "L2_SPE",
        "L2_MCC", "L2_AUC", "macroF1", "QWK", "L1_MCC_fold_sd", "L2_MCC_fold_sd", "QWK_fold_sd", "L1_MCC_CI", "L2_MCC_CI", "QWK_CI"]
T2 = T2[[c for c in cols if c in T2.columns]]
T2.to_csv(f"{OUT}/T2_cv_random_vs_blocked.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(T2.round(3).to_string(index=False))

# protocol inflation
infl = []
for mname in MODELS_MAIN:
    a = T2[(T2.protocol == "random") & (T2.model == mname)]; b = T2[(T2.protocol == "blocked") & (T2.model == mname)]
    if len(a) and len(b):
        infl.append(dict(model=mname, **{f"d_{k}": a[k].values[0] - b[k].values[0] for k in ["L1_MCC", "L1_AUC", "L2_MCC", "L2_AUC", "QWK"]}))
T3 = pd.DataFrame(infl); T3.to_csv(f"{OUT}/T3_random_minus_blocked.csv", index=False)
print("\nRandom-split minus chromosome-blocked:\n", T3.round(3).to_string(index=False))

# Independent set: legacy (contaminated) vs clean
rows = []
y_ind = ind.tier.values; clean = ind.clean.values
for proto in ["random", "blocked"]:
    for mname in MODELS_MAIN:
        ks = [k for k in RES if k[0] == proto and k[1] == mname]
        if not ks: continue
        P = np.mean([np.mean(RES[k]["ind"], 0) for k in ks], 0)
        dl = tier_metrics(P, y_ind); dc = tier_metrics(P[clean], y_ind[clean])
        rows.append(dict(protocol=proto, model=mname,
                         legacy_L1_ACC=dl["L1_ACC"], legacy_L1_MCC=dl["L1_MCC"], legacy_L1_AUC=dl["L1_AUC"],
                         legacy_L2_ACC=dl.get("L2_ACC"), legacy_L2_SEN=dl.get("L2_SEN"), legacy_L2_SPE=dl.get("L2_SPE"),
                         legacy_L2_MCC=dl.get("L2_MCC"), legacy_L2_AUC=dl.get("L2_AUC"),
                         clean_weak_vs_non_MCC=dc["L1_MCC"], clean_weak_vs_non_AUC=dc["L1_AUC"]))
T4 = pd.DataFrame(rows); T4.to_csv(f"{OUT}/T4_independent_legacy_vs_clean.csv", index=False)
print("\nIndependent set (legacy = with the 100 duplicated strong enhancers; clean = weak vs non only):\n", T4.round(3).to_string(index=False))

# Ablations (blocked)
rows = []; FOLDS_B = make_folds("blocked")
for v in ENGRADE_VARIANTS:
    ks = [k for k in RES if k[0] == "blocked" and k[1] == v and k[2] == CFG["seeds_other"][0]]   # same seed for every variant
    if not ks: continue
    P = np.mean([RES[k]["oof"] for k in ks], 0)
    for f, (_, te) in enumerate(FOLDS_B):                      # spread across held-out chromosome folds
        d = tier_metrics(P[te], y_all[te]); d.update(variant=v, fold=f); rows.append(d)
T5 = (pd.DataFrame(rows).groupby("variant")[["L1_MCC", "L1_AUC", "L2_MCC", "L2_AUC", "macroF1", "QWK"]]
      .agg(["mean", "std"]).reindex([v for v in ENGRADE_VARIANTS if any(k[1] == v for k in RES)]))
T5.columns = [f"{a}_{b}" for a, b in T5.columns]; T5.to_csv(f"{OUT}/T5_ablation_blocked.csv")
print("\nAblation (chromosome-blocked):\n", T5.round(3).to_string())

# Paired bootstrap: EnGrade vs each baseline (blocked)
PE, _ = mean_P("blocked", "EnGrade"); rows = []
for mname in MODELS_MAIN[:-1]:
    PB, ks = mean_P("blocked", mname)
    if not ks: continue
    for key in ["L1_MCC", "L2_MCC", "QWK"]:
        mdiff, ci, p = paired_boot(PE, PB, y_all, key)
        rows.append(dict(baseline=mname, metric=key, delta=mdiff, ci_lo=ci[0], ci_hi=ci[1], p_two_sided=p))
T6 = pd.DataFrame(rows); T6.to_csv(f"{OUT}/T6_paired_bootstrap.csv", index=False)
print("\nEnGrade minus baseline (paired bootstrap, blocked):\n", T6.round(4).to_string(index=False))

# %%
# ---------------------------------------------------------------- 11. Conformal evaluation
alpha = CFG["alpha"]; rows = []
for proto in ["blocked", "random"]:
    for mname in MODELS_MAIN:
        for k in [k for k in RES if k[0] == proto and k[1] == mname]:
            for mond in (True, False):
                for (f, Pc, yc, te) in RES[k]["cal"]:
                    S = conformal_sets(Pc, yc, RES[k]["oof"][te], alpha, mondrian=mond)
                    d = set_stats(S, y_all[te]); d.update(protocol=proto, model=mname, seed=k[2], fold=f,
                                                          scheme="tier-conditional" if mond else "marginal"); rows.append(d)
CP = pd.DataFrame(rows)
T7 = CP.groupby(["protocol", "model", "scheme"])[["coverage", "cov_non", "cov_weak", "cov_strong", "mean_size",
                                                   "singleton", "singleton_acc"]].mean().reset_index()
T7.to_csv(f"{OUT}/T7_conformal_alpha{alpha}.csv", index=False)
print(f"\nConformal sets at alpha={alpha} (mean over folds/seeds):\n", T7.round(3).to_string(index=False))

# coverage across alpha for EnGrade (blocked)
alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]; rows = []
for a in alphas:
    for k in [k for k in RES if k[0] == "blocked" and k[1] == "EnGrade"]:
        for mond in (True, False):
            for (f, Pc, yc, te) in RES[k]["cal"]:
                d = set_stats(conformal_sets(Pc, yc, RES[k]["oof"][te], a, mond), y_all[te])
                d.update(alpha=a, scheme="tier-conditional" if mond else "marginal"); rows.append(d)
CPA = pd.DataFrame(rows).groupby(["alpha", "scheme"]).mean(numeric_only=True).reset_index()
CPA.to_csv(f"{OUT}/T8_engrade_coverage_vs_alpha.csv", index=False)

# selective prediction: accept only singleton sets
rows = []
for mname in MODELS_MAIN:
    for k in [k for k in RES if k[0] == "blocked" and k[1] == mname]:
        for (f, Pc, yc, te) in RES[k]["cal"]:
            S = conformal_sets(Pc, yc, RES[k]["oof"][te], alpha, True); acc = S.sum(1) == 1
            yh = RES[k]["oof"][te].argmax(1)
            rows.append(dict(model=mname, accepted=acc.mean(), err_accepted=(yh[acc] != y_all[te][acc]).mean() if acc.any() else np.nan,
                             err_all=(yh != y_all[te]).mean()))
T9 = pd.DataFrame(rows).groupby("model").mean().reset_index(); T9.to_csv(f"{OUT}/T9_selective_prediction.csv", index=False)
print("\nSelective prediction (accept singleton tier-conditional sets):\n", T9.round(3).to_string(index=False))

# %%
# ---------------------------------------------------------------- 12. External shift test (Cohn, enhancer vs non)
if EXT_OK:
    y_ext = ext.label.values.astype(int); rows = []
    for mname in MODELS_MAIN:
        ks = [k for k in RES if k[0] == "blocked" and k[1] == mname]
        if not ks: continue
        Pall = [P for k in ks for P in RES[k]["ext"]]; P = np.mean(Pall, 0); s = P[:, 1] + P[:, 2]
        covs = []
        for k in ks:
            for (f, Pc, yc, te), Pe in zip(RES[k]["cal"], RES[k]["ext"]):
                S = conformal_sets(Pc, yc, Pe, alpha, True)
                hit = np.where(y_ext == 1, S[:, 1] | S[:, 2], S[:, 0]); covs.append(hit.mean())
        rows.append(dict(model=mname, AUC=roc_auc_score(y_ext, s), MCC_at_0_5=mcc_at(y_ext, s),
                         best_MCC=max(mcc_at(y_ext, s, t) for t in np.quantile(s, np.linspace(0.05, 0.95, 37))),
                         presence_coverage=np.mean(covs)))
    T10 = pd.DataFrame(rows); T10.to_csv(f"{OUT}/T10_external_cohn.csv", index=False)
    print("\nExternal shift (Genomic Benchmarks human_enhancers_cohn, zero-shot):\n", T10.round(3).to_string(index=False))

# %%
# ---------------------------------------------------------------- 12b. Robustness to point substitutions
rows = []; y_rob = y_all[ROB_IDX]
for (mname, seed, r), P in ROB_RES.items():
    d = tier_metrics(P, y_rob); rows.append(dict(model=mname, rate=r, L1_MCC=d["L1_MCC"], L1_AUC=d["L1_AUC"],
                                                 L2_MCC=d.get("L2_MCC"), L2_AUC=d.get("L2_AUC"), QWK=d["QWK"]))
T14 = pd.DataFrame(rows).groupby(["model", "rate"]).mean().reset_index()
T14.to_csv(f"{OUT}/T14_robustness_substitution.csv", index=False)
print("\nRobustness to random point substitutions (held-out-fold models, n=%d):\n" % len(ROB_IDX), T14.round(3).to_string(index=False))

# %%
# ---------------------------------------------------------------- 12c. Efficiency
def n_params(m, trainable=True): return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))
rows = []
eg = EnGrade(d_comp, d_glm, views=FULL_VIEWS); cd = CNNDLSTM()
GLM_PARAMS = 0; glm_rate = np.nan
if GLM_OK:
    from transformers import AutoModel, AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(CFG["glm_name"]); _g = AutoModel.from_pretrained(CFG["glm_name"]).to(DEVICE).eval()
    GLM_PARAMS = n_params(_g, trainable=False)
    sample = ENC_IND["seqs"][:128]
    batch = [" ".join(x[j:j + 6] for j in range(len(x) - 5)) for x in sample]
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(batch), 32):
            enc = _tok(batch[i:i + 32], return_tensors="pt", padding=True).to(DEVICE); _g(**enc)
    if DEVICE.type == "cuda": torch.cuda.synchronize()
    glm_rate = (time.time() - t0) / len(batch) * 1000; del _g
def latency(pr, reps=3):
    idx = np.resize(IDX_IND, 1000); best = 1e9
    for _ in range(reps):
        t0 = time.time(); pr(ENC_IND, idx); best = min(best, time.time() - t0)
    return best
for mname, pr in LIVE_BLOCKED.items():
    rows.append(dict(model=mname,
                     trainable_params=(n_params(eg) if mname == "EnGrade" else 2 * n_params(cd) if mname.startswith("CNN") else np.nan),
                     frozen_params=(GLM_PARAMS if ("EnGrade" == mname or "gLM" in mname) else 0),
                     train_s_per_fold=np.mean(FIT_TIME.get(mname, [np.nan])),
                     head_latency_s_per_1000=latency(pr),
                     glm_embed_s_per_1000=(glm_rate if ("EnGrade" == mname or "gLM" in mname) else 0.0)))
T15 = pd.DataFrame(rows); T15.to_csv(f"{OUT}/T15_efficiency.csv", index=False)
print("\nEfficiency (device: %s):\n" % DEVICE, T15.round(3).to_string(index=False))

# %%
# ---------------------------------------------------------------- 13. Attribution (gradient x input through the convolutional view)
TOP_KMERS = None
if ATTR_MODEL and "cnn" in FULL_VIEWS:
    m, sc, te = ATTR_MODEL["model"], ATTR_MODEL["sc"], ATTR_MODEL["te"]
    idx = te[y_all[te] == 2]
    B = batch_of(ENC_TR, idx, *sc); B["X"].requires_grad_(True)
    m.eval(); o = m(B)[:, 1].sum(); o.backward()                         # conditional logit P(strong | enhancer)
    gx = (B["X"].grad * B["X"]).detach().cpu().numpy()[:, :4].sum(1)      # (n, L)
    PROFILE = np.abs(gx).mean(0)
    from collections import Counter
    hits = Counter()
    for r, i in enumerate(idx):
        seq = ENC_TR["seqs"][i]; sm = np.convolve(gx[r], np.ones(8), "valid")
        for p in np.argsort(-sm)[:3]:
            w = seq[p:p + 8]
            for j in range(3): hits[w[j:j + 6]] += 1
    bg = Counter(); non_idx = np.where(y_all == 0)[0]
    for i in non_idx:
        s = ENC_TR["seqs"][i]
        for j in range(len(s) - 5): bg[s[j:j + 6]] += 1
    tot_h, tot_b = sum(hits.values()), sum(bg.values())
    rows = [dict(kmer=k, hits=c, log2_enrichment=np.log2((c / tot_h) / ((bg.get(k, 0) + 1) / tot_b)))
            for k, c in hits.items() if c >= 4 and "N" not in k]
    if rows:
        TOP_KMERS = pd.DataFrame(rows).sort_values("log2_enrichment", ascending=False).head(20)
        TOP_KMERS.to_csv(f"{OUT}/T11_attributed_6mers_strong.csv", index=False)
        print("\nTop attributed 6-mers in strong enhancers (vs non-enhancer background):\n", TOP_KMERS.round(2).to_string(index=False))

# %%
# ---------------------------------------------------------------- 13b. Prediction-set composition (EnGrade, blocked)
comp_rows = []
for k in [k for k in RES if k[0] == "blocked" and k[1] == "EnGrade"]:
    for (f, Pc, yc, te) in RES[k]["cal"]:
        S = conformal_sets(Pc, yc, RES[k]["oof"][te], CFG["alpha"], True)
        for row, yy in zip(S, y_all[te]): comp_rows.append(dict(tier=yy, set="".join("NWS"[j] for j in range(3) if row[j])))
SC = pd.crosstab(pd.DataFrame(comp_rows).tier, pd.DataFrame(comp_rows).set, normalize="index")
SC.to_csv(f"{OUT}/T13_engrade_set_composition.csv"); print(SC.round(3))
if GATES:
    GD_ = pd.concat(GATES); GD_.groupby("tier")[[c for c in ("cnn", "comp", "glm") if c in GD_.columns]].mean().to_csv(f"{OUT}/T12_gate_weights_by_tier.csv")

# %%
# ---------------------------------------------------------------- 13c. Per-fold results and confusion matrix (from saved predictions)
import pickle
from sklearn.metrics import confusion_matrix
_D = pickle.load(open(f"{OUT}/predictions.pkl", "rb")); _y = _D["y_all"]; _ch = _D["chrom"]
_nf = json.load(open(f"{OUT}/config.json"))["n_folds"]
_folds = list(StratifiedGroupKFold(_nf, shuffle=True, random_state=0).split(np.zeros(len(_y)), _y, groups=_ch))
def _mp(m): return np.mean([v["oof"] for k, v in _D["RES"].items() if k[0] == "blocked" and k[1] == m], 0)
rows = []
for f_, (_, te_) in enumerate(_folds):
    for m_ in ["StackFRL (multiview stacking)", "EnGrade"]:
        d_ = tier_metrics(_mp(m_)[te_], _y[te_])
        rows.append(dict(fold=f_ + 1, chromosomes=" ".join(sorted(set(_ch[te_]), key=lambda c: (len(c), c))).replace("chr", "").upper(),
                         n=len(te_), model=m_, L1_MCC=d_["L1_MCC"], L2_MCC=d_.get("L2_MCC"), QWK=d_["QWK"]))
T16 = pd.DataFrame(rows); T16.to_csv(f"{OUT}/T16_per_fold.csv", index=False); print(T16.round(3).to_string(index=False))
CM = confusion_matrix(_y, _mp("EnGrade").argmax(1), labels=[0, 1, 2])
pd.DataFrame(CM, index=["non", "weak", "strong"], columns=["pred_non", "pred_weak", "pred_strong"]).to_csv(f"{OUT}/T17_confusion_engrade.csv")
print(CM)

# %%
# ---------------------------------------------------------------- 14. Figures (black-and-white, journal widths)
# Every figure below is rebuilt from engrade_outputs/predictions.pkl and the CSV tables, so it can be re-run alone.
import pickle, matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
D = pickle.load(open(f"{OUT}/predictions.pkl", "rb"))
RES_, y_all_ = D["RES"], D["y_all"]
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 600, "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                     "legend.frameon": False, "legend.fontsize": 7})
GREY = ["#000000", "#4d4d4d", "#8c8c8c", "#c8c8c8"]
HATCH = ["", "////", "....", "xxxx", "\\\\\\\\", "++", "oo"]
LS = ["-", "--", "-.", ":"]; MK = ["o", "s", "^", "D"]
W1, W2 = 3.5, 7.16
short = {"StackFRL (multiview stacking)": "StackFRL", "CNN-DLSTM cascade": "CNN-DLSTM",
         "gLM probe (DNABERT-6 + LR)": "gLM probe", "EnGrade": "EnGrade-CP"}
MODELS_F = [m for m in short if any(k[1] == m for k in RES_)]
def meanP(proto, m):
    ks = [k for k in RES_ if k[0] == proto and k[1] == m]; return np.mean([RES_[k]["oof"] for k in ks], 0)
T2f = pd.read_csv(f"{OUT}/T2_cv_random_vs_blocked.csv"); T4f = pd.read_csv(f"{OUT}/T4_independent_legacy_vs_clean.csv")
def bars(ax, x, h, i, w, **kw):
    return ax.bar(x, h, w, color="white" if i % 2 else GREY[min(i, 3)], edgecolor="black", hatch=HATCH[i], linewidth=0.7, **kw)

# Fig. 2: leakage audit
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.35), gridspec_kw=dict(width_ratios=[1, 1, 1.35], wspace=0.45))
lab = ["strong", "weak", "non"]; tot = [100, 100, 200]
dup = [int(np.sum(~D["clean"][D["y_ind"] == k])) for k in (2, 1, 0)]
ax[0].bar(lab, tot, 0.6, color="white", edgecolor="black", linewidth=0.7, label="independent set")
ax[0].bar(lab, dup, 0.6, color=GREY[1], edgecolor="black", linewidth=0.7, label="exact copy in training")
ax[0].set_ylabel("sequences"); ax[0].set_ylim(0, 260); ax[0].legend(loc="upper left"); ax[0].set_title("(a)", loc="left", fontweight="bold")
ls_ = pd.Series(D["loc_sizes"]).clip(upper=10); vc = ls_.value_counts().sort_index()
ax[1].bar([str(i) if i < 10 else "≥10" for i in vc.index], vc.values, 0.6, color=GREY[2], edgecolor="black", linewidth=0.7)
ax[1].tick_params(axis="x", labelsize=7)
ax[1].set_yscale("log"); ax[1].set_xlabel("tiles per abutting locus"); ax[1].set_ylabel("loci (log scale)"); ax[1].set_title("(b)", loc="left", fontweight="bold")
ms = MODELS_F; x = np.arange(len(ms))
cv = [T2f[(T2f.protocol == "blocked") & (T2f.model == m)].L2_MCC.values[0] for m in ms]
lg = [T4f[(T4f.protocol == "blocked") & (T4f.model == m)].legacy_L2_MCC.values[0] for m in ms]
bars(ax[2], x - 0.19, cv, 0, 0.36, label="chromosome-blocked CV")
bars(ax[2], x + 0.19, lg, 1, 0.36, label="legacy independent set")
ax[2].set_xticks(x); ax[2].set_xticklabels([short[m] for m in ms], rotation=30, ha="right", fontsize=7)
ax[2].set_ylabel("strong vs weak MCC"); ax[2].set_ylim(0, 0.8); ax[2].legend(loc="upper left"); ax[2].set_title("(c)", loc="left", fontweight="bold")
fig.savefig(f"{OUT}/Fig2_leakage_audit.png", bbox_inches="tight"); plt.close(fig)

# Fig. 3: random vs blocked
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3), gridspec_kw=dict(wspace=0.35))
for a, key, tt in zip(ax, ["L1_MCC", "L2_MCC", "QWK"], ["(a) enhancer vs non, MCC", "(b) strong vs weak, MCC", "(c) three-tier QWK"]):
    r = [T2f[(T2f.model == m) & (T2f.protocol == "random")][key].values[0] for m in ms]
    b = [T2f[(T2f.model == m) & (T2f.protocol == "blocked")][key].values[0] for m in ms]
    bars(a, x - 0.19, r, 1, 0.36, label="random split"); bars(a, x + 0.19, b, 0, 0.36, label="chromosome-blocked")
    a.set_xticks(x); a.set_xticklabels([short[m] for m in ms], rotation=25, ha="right"); a.set_title(tt, loc="left", fontsize=8)
    a.set_ylim(0, max(max(r), max(b)) * 1.25)
h, l = ax[0].get_legend_handles_labels(); fig.legend(h, l, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.08))
fig.savefig(f"{OUT}/Fig3_random_vs_blocked.png", bbox_inches="tight"); plt.close(fig)

# Fig. 4: ROC curves (blocked)
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.8), gridspec_kw=dict(wspace=0.45)); e = y_all_ > 0
for i, m in enumerate(ms):
    P = meanP("blocked", m); s1 = P[:, 1] + P[:, 2]
    f1, t1, _ = roc_curve(e, s1); ax[0].plot(f1, t1, LS[i], c="black", lw=1.1 if m == "EnGrade" else 0.9, label=f"{short[m]} (AUC {roc_auc_score(e, s1):.3f})")
    s2 = P[e, 2] / np.clip(P[e, 1] + P[e, 2], 1e-9, None); yy = y_all_[e] == 2
    f2, t2, _ = roc_curve(yy, s2); ax[1].plot(f2, t2, LS[i], c="black", lw=1.1 if m == "EnGrade" else 0.9, label=f"{short[m]} (AUC {roc_auc_score(yy, s2):.3f})")
for a, tt in zip(ax, ["(a) enhancer vs non-enhancer", "(b) strong vs weak enhancer"]):
    a.plot([0, 1], [0, 1], c=GREY[2], lw=0.6); a.set_xlabel("false positive rate"); a.set_ylabel("true positive rate")
    a.set_title(tt, loc="left", fontsize=8); a.set_aspect("equal")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, handlelength=2.6, frameon=False)
fig.savefig(f"{OUT}/Fig4_roc_blocked.png", bbox_inches="tight"); plt.close(fig)

# Fig. 5: coverage vs alpha
CPAf = pd.read_csv(f"{OUT}/T8_engrade_coverage_vs_alpha.csv")
fig, ax = plt.subplots(1, 3, figsize=(W2, 2.35), gridspec_kw=dict(wspace=0.38))
for a, sch, tt in zip(ax[:2], ["marginal", "tier-conditional"], ["(a) marginal calibration", "(b) tier-conditional calibration"]):
    d = CPAf[CPAf.scheme == sch]
    for i, nm in enumerate(["non", "weak", "strong"]): a.plot(d.alpha, d[f"cov_{nm}"], LS[i], marker=MK[i], ms=3.2, c="black", mfc="white", lw=0.9, label=nm)
    a.plot(d.alpha, 1 - d.alpha, c=GREY[2], lw=2.2, alpha=0.6, label="target", zorder=0)
    a.set_xlabel(r"miscoverage level $\alpha$"); a.set_ylabel("per-tier coverage"); a.set_ylim(0.2, 1.02); a.set_title(tt, loc="left", fontsize=8)
ax[0].legend(loc="lower left")
for i, sch in enumerate(["marginal", "tier-conditional"]):
    d = CPAf[CPAf.scheme == sch]; ax[2].plot(d.alpha, d.mean_size, LS[i], marker=MK[i], ms=3.2, c="black", mfc="white", lw=0.9, label=sch)
ax[2].set_xlabel(r"miscoverage level $\alpha$"); ax[2].set_ylabel("mean set size"); ax[2].set_title("(c) set size", loc="left", fontsize=8); ax[2].legend(loc="upper right")
fig.savefig(f"{OUT}/Fig5_coverage_vs_alpha.png", bbox_inches="tight"); plt.close(fig)

# Fig. 6: set composition by true tier
SCf = pd.read_csv(f"{OUT}/T13_engrade_set_composition.csv", index_col=0)
order = [c for c in ["N", "NW", "W", "WS", "S", "NWS"] if c in SCf.columns]
fig, ax = plt.subplots(figsize=(W1 + 0.6, 1.9)); left = np.zeros(len(SCf))
fills = {"N": "white", "NW": GREY[3], "W": "white", "WS": GREY[2], "S": GREY[1], "NWS": "white"}
hat = {"N": "", "NW": "", "W": "....", "WS": "", "S": "", "NWS": "////"}
for c in order:
    ax.barh(["non", "weak", "strong"][:len(SCf)], SCf[c].values, left=left, color=fills[c], hatch=hat[c], edgecolor="black", lw=0.6,
            label="{" + ",".join(c) + "}"); left += SCf[c].values
ax.set_xlabel(r"share of held-out sequences ($\alpha$ = %.2f)" % CFG["alpha"]); ax.set_xlim(0, 1)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="prediction set", title_fontsize=7)
fig.savefig(f"{OUT}/Fig6_set_composition.png", bbox_inches="tight"); plt.close(fig)

# Fig. 7: ablation
T5f = pd.read_csv(f"{OUT}/T5_ablation_blocked.csv", index_col=0)
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw=dict(wspace=0.08), sharey=True); yv = np.arange(len(T5f))[::-1]
for a, key, tt in zip(ax, ["L1_MCC", "QWK"], ["(a) enhancer vs non, MCC", "(b) three-tier QWK"]):
    a.barh(yv, T5f[f"{key}_mean"], xerr=T5f[f"{key}_std"], color=[GREY[1] if i == 0 else "white" for i in range(len(T5f))],
           edgecolor="black", lw=0.7, capsize=2, error_kw=dict(lw=0.7))
    a.axvline(T5f[f"{key}_mean"].iloc[0], c="black", ls=":", lw=0.8); a.set_title(tt, loc="left", fontsize=8)
    lo = (T5f[f"{key}_mean"] - T5f[f"{key}_std"]).min(); a.set_xlim(max(0, lo - 0.05), (T5f[f"{key}_mean"] + T5f[f"{key}_std"]).max() + 0.02)
ax[0].set_yticks(yv); ax[0].set_yticklabels(["EnGrade-CP (full)"] + T5f.index[1:].tolist())
fig.savefig(f"{OUT}/Fig7_ablation.png", bbox_inches="tight"); plt.close(fig)

# Fig. 8: robustness to substitutions
T14f = pd.read_csv(f"{OUT}/T14_robustness_substitution.csv")
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.35), gridspec_kw=dict(wspace=0.3))
for a, key, tt in zip(ax, ["L1_MCC", "L2_MCC"], ["(a) enhancer vs non, MCC", "(b) strong vs weak, MCC"]):
    for i, m in enumerate(ms):
        d = T14f[T14f.model == m].sort_values("rate")
        a.plot(d.rate * 100, d[key], LS[i], marker=MK[i], ms=3.5, c="black", mfc="white" if m != "EnGrade" else "black", lw=0.9, label=short[m])
    a.set_xlabel("substituted positions (%)"); a.set_title(tt, loc="left", fontsize=8); a.set_xticks([0, 2, 5, 10])
ax[0].set_ylabel("MCC"); h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, handlelength=2.6)
fig.savefig(f"{OUT}/Fig8_robustness.png", bbox_inches="tight"); plt.close(fig)

# Fig. 9: external shift
if D["y_ext"] is not None:
    y_ext_ = D["y_ext"]; T10f = pd.read_csv(f"{OUT}/T10_external_cohn.csv")
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.6), gridspec_kw=dict(wspace=0.35, width_ratios=[1, 1.1]))
    for i, m in enumerate(ms):
        ks = [k for k in RES_ if k[0] == "blocked" and k[1] == m]; P = np.mean([P for k in ks for P in RES_[k]["ext"]], 0); s = P[:, 1] + P[:, 2]
        f_, t_, _ = roc_curve(y_ext_, s); ax[0].plot(f_, t_, LS[i], c="black", lw=1.1 if m == "EnGrade" else 0.9, label=f"{short[m]} ({roc_auc_score(y_ext_, s):.3f})")
    ax[0].plot([0, 1], [0, 1], c=GREY[2], lw=0.6); ax[0].set_xlabel("false positive rate"); ax[0].set_ylabel("true positive rate")
    ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, title="model (AUC)", title_fontsize=7, handlelength=2.6)
    ax[0].set_title("(a) zero-shot ROC", loc="left", fontsize=8); ax[0].set_aspect("equal")
    xx = np.arange(len(ms)); cov = [T10f[T10f.model == m].presence_coverage.values[0] for m in ms]
    for xi, c, m in zip(xx, cov, ms): ax[1].bar([xi], [c], 0.55, color=GREY[1] if m == "EnGrade" else "white", edgecolor="black", lw=0.7)
    ax[1].axhline(1 - CFG["alpha"], c="black", ls="--", lw=0.8); ax[1].text(len(ms) - 0.5, 1 - CFG["alpha"] + 0.004, "target", ha="right", va="bottom", fontsize=7)
    ax[1].set_ylim(0.8, 0.92); ax[1].set_xticks(xx); ax[1].set_xticklabels([short[m] for m in ms], rotation=20, ha="right")
    ax[1].set_ylabel("presence coverage"); ax[1].set_title("(b) coverage under shift", loc="left", fontsize=8)
    fig.savefig(f"{OUT}/Fig9_external_shift.png", bbox_inches="tight"); plt.close(fig)

# Fig. 10: gate weights and attributed 6-mers
GD = pd.concat(D["GATES"]); vcols = [c for c in ["cnn", "comp", "glm"] if c in GD.columns]
fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5), gridspec_kw=dict(wspace=0.45, width_ratios=[1, 1.2]))
xx = np.arange(3); names = {"cnn": "convolutional", "comp": "composition", "glm": "gLM"}
for i, v in enumerate(vcols):
    mu = [GD[GD.tier == k][v].mean() for k in range(3)]; sd = [GD[GD.tier == k][v].std() for k in range(3)]
    lo = [m_ - max(m_ - s_, 0.0) for m_, s_ in zip(mu, sd)]; hi = [min(m_ + s_, 1.0) - m_ for m_, s_ in zip(mu, sd)]
    bars(ax[0], xx + (i - 1) * 0.26, mu, i, 0.25, yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.6), label=names[v])
ax[0].set_xticks(xx); ax[0].set_xticklabels(["non", "weak", "strong"]); ax[0].set_ylabel("gate weight"); ax[0].set_ylim(0, 1.05)
ax[0].legend(loc="upper left", ncol=3, bbox_to_anchor=(0, 1.13)); ax[0].set_title("(a)", loc="left", fontweight="bold", y=1.12)
if os.path.exists(f"{OUT}/T11_attributed_6mers_strong.csv"):
    tk = pd.read_csv(f"{OUT}/T11_attributed_6mers_strong.csv").head(12)[::-1]
    ax[1].barh(tk.kmer, tk.log2_enrichment, color=GREY[2], edgecolor="black", lw=0.6)
    ax[1].set_xlabel("log2 enrichment vs non-enhancers"); ax[1].tick_params(axis="y", labelsize=6.5)
    for lbl in ax[1].get_yticklabels(): lbl.set_fontfamily("monospace")
ax[1].set_title("(b)", loc="left", fontweight="bold", y=1.12)
fig.savefig(f"{OUT}/Fig10_gate_attribution.png", bbox_inches="tight"); plt.close(fig)

# Fig. 1: architecture schematic. Solid = forward pass; dashed = reverse-mode gradient of the training loss,
# drawn antiparallel to each trainable forward edge; no gradient enters the frozen DNABERT-6 encoder.
fig, ax = plt.subplots(figsize=(W2, 3.7)); ax.set_xlim(0, 100); ax.set_ylim(-3, 58); ax.axis("off")
FS = 6.2
def box(x0, y0, x1, y1, txt, fill="white", lw=0.8, hatch=""):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0.0,rounding_size=0.9",
                                fc=fill, ec="black", lw=lw, hatch=hatch))
    ax.text((x0 + x1) / 2, (y0 + y1) / 2, txt, ha="center", va="center", fontsize=FS, linespacing=1.3)
    return dict(x0=x0, y0=y0, x1=x1, y1=y1, xm=(x0 + x1) / 2, ym=(y0 + y1) / 2)
def arr(p, q, dashed=False):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.85 if not dashed else 0.9,
                                 ls=(0, (3.2, 2.0)) if dashed else "-", color="black", shrinkA=0, shrinkB=0))
OFF = 1.1   # vertical offset separating forward (above) and backward (below) lanes
inp = box(0.3, 22, 9.3, 33, "200-bp\nsequence\n$s$")
v1 = box(12, 41, 31.8, 52, "one-hot, NCP, EIIP\nchannels $X$ and\nreverse complement $\\bar{X}$")
v2 = box(12, 22, 31.8, 33, "composition\nvector $h$: Kmer,\nCKSNAP, PseEIIP, PAC")
v3 = box(12, 3, 31.8, 14, "overlapping 6-mers\nDNABERT-6\n(frozen)", fill=GREY[3])
e1 = box(35, 41, 49, 52, "dilated CNN $f_\\theta$\nstrand-averaged\n$z_1$")
e2 = box(35, 22, 49, 33, "MLP $g_\\phi$\n$z_2$")
e3 = box(35, 3, 49, 14, "mean pool +\nMLP $\\psi_\\omega$\n$z_3$")
gate = box(54, 3, 67, 52, "softmax gate\n$\\pi_v(s)$\n\nview dropout\n(training)\n\n$z=\\sum_v \\pi_v z_v$")
head = box(72, 22, 84, 33, "CORN head\n$\\ell_1$, $\\ell_2$")
loss = box(72, 41, 84, 52, "training loss\n$\\mathcal{L}_{\\mathrm{CORN}}$", lw=0.8)
prob = box(88, 22, 99.5, 33, "tier\nprobabilities\n$p_0, p_1, p_2$")
cal = box(88, 3, 99.5, 14, "per-tier\nthresholds $\\hat{q}_k$\n(calibration)")
outb = box(71, 3, 84.5, 14, "contiguous set\n$\\Gamma_\\alpha(s)$", lw=1.2)
# forward pass
for v in (v1, v2, v3): arr((inp["x1"], inp["ym"]), (v["x0"], v["ym"]))
for v, e in ((v1, e1), (v2, e2), (v3, e3)): arr((v["x1"], v["ym"]), (e["x0"], e["ym"]))
for e in (e1, e2, e3): arr((e["x1"], e["ym"] + OFF), (gate["x0"], e["ym"] + OFF))
arr((gate["x1"], head["ym"] + OFF), (head["x0"], head["ym"] + OFF))
arr((head["xm"] - 2.2, head["y1"]), (head["xm"] - 2.2, loss["y0"]))
arr((head["x1"], head["ym"]), (prob["x0"], prob["ym"]))
arr((prob["xm"], prob["y0"]), (prob["xm"], cal["y1"]))
arr((cal["x0"], cal["ym"]), (outb["x1"], outb["ym"]))
# reverse-mode gradient (training only)
arr((loss["xm"] + 2.2, loss["y0"]), (loss["xm"] + 2.2, head["y1"]), dashed=True)
arr((head["x0"], head["ym"] - OFF), (gate["x1"], head["ym"] - OFF), dashed=True)
for e in (e1, e2, e3): arr((gate["x0"], e["ym"] - OFF), (e["x1"], e["ym"] - OFF), dashed=True)
ax.text(22, -1.8, "shaded encoder is frozen: no gradient enters DNABERT-6", fontsize=5.6, style="italic", ha="center", va="bottom")
ax.plot([], [], c="black", lw=0.85, label="forward pass")
ax.plot([], [], c="black", lw=0.9, ls=(0, (3.2, 2.0)), label="reverse-mode gradient (training only)")
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, fontsize=6.4, handlelength=3.0)
fig.savefig(f"{OUT}/Fig1_architecture.png", bbox_inches="tight"); plt.close(fig)
print("figures written:", sorted(f for f in os.listdir(OUT) if f.startswith("Fig")))

# %%
# ---------------------------------------------------------------- 15. Summary
summary = dict(
    device=str(DEVICE), glm_view=GLM_OK, external=EXT_OK, runtime_min=round((time.time() - T0) / 60, 1),
    duplicated_independent=int(ind.dup_in_train.sum()), abutting_tiles=int(abut),
    blocked=T2[T2.protocol == "blocked"].set_index("model")[["L1_MCC", "L1_AUC", "L2_MCC", "L2_AUC", "QWK"]].round(4).to_dict("index"),
    random=T2[T2.protocol == "random"].set_index("model")[["L1_MCC", "L1_AUC", "L2_MCC", "L2_AUC", "QWK"]].round(4).to_dict("index"),
)
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=2, default=float)
print(json.dumps(summary, indent=2, default=float))
print("\nAll tables and figures are in", os.path.abspath(OUT))
try:
    shutil.make_archive("engrade_outputs", "zip", OUT); print("Zipped to engrade_outputs.zip")
    from google.colab import files  # noqa: F401  (download manually from the Files pane if preferred)
except Exception:
    pass
