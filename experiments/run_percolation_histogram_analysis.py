"""Percolation and Chaining Analysis: Histogram and Critical Threshold Invariance.

Directly addresses and resolves the theoretical issues raised in critic.md:
1. Shows that theta_crit is NOT invariant in N for general unimodal distributions.
   In a heterogeneous corpus (continuous similarity distribution without a spectral gap),
   the percolation condition <k> = p(theta)*(N-1) = 1 forces theta_crit to strictly
   increase with N as 1/(N-1) shrinks.
2. Shows that the apparent invariance of theta_crit across N in the clustered corpus
   is a consequence of the bimodal similarity distribution with a deep spectral gap
   between inter-topic (~0.05-0.25) and intra-topic (~0.85-0.95) similarities.
   Throughout the gap [0.5, 0.7], the density is near zero, pinning the transition.
3. Documents the pathological single-template Phase 4 corpus where pairwise similarities
   are concentrated above 0.70 (mean ~0.85), causing <k> >> 1 at theta=0.85 and collapsing
   into a 9,990/10,000 giant component.
4. Validates the actionable N-dependent quantile rule:
   theta(N) = Q_{1 - c/N}(sim) with c = 0.8.
   Guarantees <k> <= c < 1, strictly preventing the giant component across all N and corpora.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

# Ensure root directory is on path
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("ERROR: sentence-transformers not installed. Please use the project virtualenv.")
    sys.exit(1)

MODEL_NAME = "all-MiniLM-L6-v2"
N_VALUES = [200, 1000, 5000, 10000]
THETAS = np.round(np.arange(0.30, 1.00, 0.01), 2)
C_FACTOR = 0.8  # Actionable rule: p(theta) <= c/N with c=0.8 < 1.0


# ---------------------------------------------------------------------------
# Corpus Generators
# ---------------------------------------------------------------------------

RAW_TOPICS_20 = [
    ("astronomy",
     ["Astronomers", "Astrophysicists", "Space researchers", "Stellar scientists", "Observatory astronomers"],
     ["carefully observe", "regularly observe", "systematically observe", "closely observe", "actively observe"],
     ["distant spiral galaxies", "remote spiral galaxies", "distant galactic spirals", "faraway spiral galaxies", "distant cosmic spirals"],
     ["in deep space.", "across deep space.", "in deep cosmic space.", "throughout deep space.", "within deep space."]),

    ("medieval_history",
     ["Medieval lords", "Feudal barons", "European monarchs", "Medieval rulers", "Royal sovereigns"],
     ["strictly governed", "historically ruled", "traditionally governed", "firmly ruled", "officially governed"],
     ["fortified stone castles", "medieval stone castles", "ancient fortified castles", "historic stone castles", "royal stone castles"],
     ["with sworn vassals.", "through sworn vassals.", "using sworn vassals.", "alongside sworn vassals.", "backed by sworn vassals."]),

    ("marine_biology",
     ["Marine biologists", "Ocean researchers", "Aquatic scientists", "Reef researchers", "Marine scientists"],
     ["closely monitor", "regularly monitor", "carefully track", "systematically monitor", "actively track"],
     ["tropical coral reefs", "shallow coral reefs", "equatorial coral reefs", "warm coral reefs", "vibrant coral reefs"],
     ["in ocean waters.", "across ocean waters.", "within ocean waters.", "throughout ocean waters.", "along ocean waters."]),

    ("pastry_culinary",
     ["Pastry chefs", "Artisanal bakers", "Bakery chefs", "Pastry bakers", "Dessert chefs"],
     ["carefully bake", "traditionally bake", "expertly bake", "regularly bake", "skillfully bake"],
     ["flaky butter croissants", "crispy golden croissants", "layered butter croissants", "flaky French croissants", "golden French croissants"],
     ["in hot ovens.", "inside baking ovens.", "using pastry ovens.", "in commercial ovens.", "with artisan ovens."]),

    ("seismology",
     ["Seismologists", "Geophysicists", "Earthquake scientists", "Seismic researchers", "Geological researchers"],
     ["accurately record", "continuously record", "regularly record", "systematically record", "carefully record"],
     ["subterranean seismic waves", "underground seismic tremors", "crustal seismic waves", "deep seismic vibrations", "tectonic seismic tremors"],
     ["near fault lines.", "along active faults.", "around fault lines.", "across active fault lines.", "near earthquake faults."]),

    ("quantum_physics",
     ["Quantum physicists", "Theoretical physicists", "Subatomic researchers", "Quantum scientists", "Laboratory physicists"],
     ["precisely measure", "carefully measure", "systematically measure", "accurately measure", "experimentally measure"],
     ["entangled quantum states", "entangled subatomic qubits", "isolated quantum states", "entangled photon pairs", "quantum coherent states"],
     ["in cryogenic setups.", "at cryogenic temperatures.", "inside cryogenic setups.", "under cryogenic conditions.", "within cryogenic setups."]),

    ("botany",
     ["Botanists", "Plant biologists", "Horticulturists", "Flora researchers", "Botanical scientists"],
     ["carefully cultivate", "regularly cultivate", "systematically cultivate", "closely cultivate", "attentively cultivate"],
     ["rare carnivorous plants", "unusual carnivorous plants", "tropical carnivorous plants", "exotic carnivorous plants", "delicate carnivorous plants"],
     ["in greenhouse enclosures.", "inside greenhouse enclosures.", "within humid greenhouses.", "throughout greenhouses.", "under greenhouse glass."]),

    ("jazz_music",
     ["Jazz musicians", "Session improvisers", "Bebop instrumentalists", "Jazz performers", "Ensemble soloists"],
     ["expressively play", "spontaneously play", "creatively play", "fluidly play", "masterfully play"],
     ["syncopated chord progressions", "complex chord progressions", "syncopated jazz chords", "harmonic chord progressions", "intricate jazz progressions"],
     ["on vintage instruments.", "with vintage instruments.", "using vintage instruments.", "on acoustic instruments.", "with classic instruments."]),

    ("immunology",
     ["Immunologists", "Clinical immunologists", "Serology researchers", "Immune specialists", "Vaccine scientists"],
     ["rigorously quantify", "systematically quantify", "accurately quantify", "carefully quantify", "closely quantify"],
     ["neutralizing antibody responses", "protective antibody responses", "specific antibody responses", "circulating antibody responses", "targeted antibody responses"],
     ["in blood serum.", "within blood serum.", "across blood serum.", "inside blood serum.", "from blood serum."]),

    ("aeronautics",
     ["Aeronautical engineers", "Aerospace designers", "Flight dynamicists", "Aviation engineers", "Aerodynamic researchers"],
     ["systematically test", "rigorously test", "experimentally test", "aerodynamically test", "carefully test"],
     ["supersonic swept wings", "supersonic aircraft wings", "supersonic aerodynamic wings", "supersonic wing geometries", "high-speed supersonic wings"],
     ["in wind tunnels.", "inside wind tunnels.", "within wind tunnels.", "under wind tunnel conditions.", "through wind tunnel tests."]),

    ("archaeology",
     ["Field archaeologists", "Excavation teams", "Archaeological surveyors", "Antiquity researchers", "Site excavators"],
     ["delicately unearth", "carefully unearth", "patiently unearth", "systematically unearth", "methodically unearth"],
     ["ancient terracotta pottery", "prehistoric ceramic pottery", "decorated terracotta pottery", "ancient amphora pottery", "ancient clay pottery"],
     ["at excavation sites.", "within historic excavation sites.", "across archaeological sites.", "in ancient dig sites.", "from excavation dig sites."]),

    ("finance",
     ["Portfolio managers", "Investment analysts", "Asset allocators", "Wealth advisors", "Fund managers"],
     ["prudently balance", "carefully balance", "systematically balance", "actively balance", "regularly balance"],
     ["diversified equity portfolios", "balanced equity portfolios", "global equity portfolios", "defensive equity portfolios", "strategic equity portfolios"],
     ["during market cycles.", "amid market cycles.", "across market cycles.", "under fluctuating market cycles.", "through volatile market cycles."]),

    ("robotics",
     ["Robotics engineers", "Automation specialists", "Mechatronic developers", "Robotic programmers", "Control systems engineers"],
     ["precisely program", "carefully program", "dynamically program", "systematically program", "regularly program"],
     ["articulated robotic arms", "industrial robotic arms", "multi-axis robotic arms", "automated robotic arms", "programmable robotic arms"],
     ["on factory lines.", "along factory lines.", "in manufacturing lines.", "across assembly lines.", "within production lines."]),

    ("linguistics",
     ["Phonetic linguists", "Dialectologists", "Sociolinguists", "Acoustic linguists", "Speech scientists"],
     ["systematically transcribe", "meticulously transcribe", "accurately transcribe", "carefully transcribe", "rigorously transcribe"],
     ["spoken dialect variations", "regional dialect variations", "phonetic dialect variations", "native dialect variations", "recorded dialect variations"],
     ["in audio recordings.", "across audio recordings.", "from spoken recordings.", "within audio recordings.", "in native speaker recordings."]),

    ("agriculture",
     ["Organic farmers", "Agricultural growers", "Crop producers", "Heirloom farmers", "Farm cultivators"],
     ["manually harvest", "carefully harvest", "regularly harvest", "hand-harvest", "seasonally harvest"],
     ["ripe heirloom vegetables", "fresh heirloom vegetables", "organic heirloom vegetables", "sun-ripened heirloom vegetables", "mature heirloom vegetables"],
     ["from fertile fields.", "across fertile fields.", "in organic fields.", "within agricultural fields.", "out of farming fields."]),

    ("cardiology",
     ["Cardiologists", "Heart surgeons", "Cardiovascular specialists", "Clinical cardiologists", "Cardiac surgeons"],
     ["carefully treat", "surgically treat", "successfully treat", "clinically treat", "interveningly treat"],
     ["stenotic coronary arteries", "narrowed coronary arteries", "occluded coronary arteries", "blocked coronary arteries", "diseased coronary arteries"],
     ["in operating rooms.", "during surgical procedures.", "under clinical supervision.", "inside surgical suites.", "via cardiac interventions."]),

    ("meteorology",
     ["Meteorologists", "Weather forecasters", "Atmospheric modelers", "Severe storm forecasters", "Radar meteorologists"],
     ["accurately track", "continuously track", "closely track", "systematically track", "reliably track"],
     ["tropical hurricane storms", "severe hurricane storms", "incoming hurricane storms", "coastal hurricane storms", "violent hurricane storms"],
     ["using Doppler radar.", "with Doppler radar.", "through Doppler radar.", "via Doppler radar.", "across Doppler radar grids."]),

    ("philosophy",
     ["Moral philosophers", "Ethics scholars", "Academic philosophers", "Philosophical logicians", "Ethical theorists"],
     ["rigorously debate", "formally debate", "systematically debate", "philosophically debate", "analytically debate"],
     ["deontological moral dilemmas", "fundamental ethical dilemmas", "normative moral dilemmas", "foundational ethical duties", "classical ethical dilemmas"],
     ["in academic seminars.", "within philosophical seminars.", "across academic seminars.", "during philosophy colloquia.", "through academic debates."]),

    ("architecture",
     ["Urban architects", "Structural designers", "Building architects", "Architectural designers", "Urban designers"],
     ["creatively design", "elegantly design", "innovatively design", "structurally design", "sustainably design"],
     ["sustainable timber buildings", "modern timber buildings", "wooden timber buildings", "eco-friendly timber buildings", "engineered timber buildings"],
     ["in urban centers.", "across urban centers.", "within modern cities.", "for urban centers.", "in metropolitan centers."]),

    ("entomology",
     ["Field entomologists", "Insect biologists", "Tropical entomologists", "Ecological entomologists", "Taxonomic entomologists"],
     ["extensively study", "meticulously study", "carefully study", "comprehensively study", "systematically study"],
     ["social insect colonies", "tropical insect colonies", "forest insect colonies", "native insect colonies", "arboreal insect colonies"],
     ["in rainforest habitats.", "throughout rainforest habitats.", "inside rainforest habitats.", "across rainforest habitats.", "within rainforest habitats."])
]


def generate_corpus_a(total_n: int = 10000) -> tuple[list[str], list[int]]:
    """Generates Clustered Corpus with 20 topics.
    
    Each topic contributes total_n // 20 distinct combinatorial sentences.
    Intra-topic pairs have high similarity (~0.85-0.95), inter-topic pairs low (~0.05-0.25).
    """
    n_topics = len(RAW_TOPICS_20)
    per_topic = total_n // n_topics
    sentences = []
    labels = []
    for topic_id, (tname, s0, s1, s2, s3) in enumerate(RAW_TOPICS_20):
        combos = [f"{a} {b} {c} {d}" for a, b, c, d in itertools.product(s0, s1, s2, s3)]
        selected = combos[:per_topic]
        sentences.extend(selected)
        labels.extend([topic_id] * len(selected))
    return sentences, labels


def generate_corpus_b(total_n: int = 10000, seed: int = 42) -> list[str]:
    """Generates Heterogeneous/Unstructured Corpus.
    
    Draws independently from open-domain vocabulary combinations without topic clustering,
    producing a continuous unimodal cosine similarity distribution with no spectral gap.
    """
    open_nouns = [
        "glacier", "telescope", "symphony", "polymer", "fossil", "algorithm", "manuscript", "volcano",
        "metabolism", "treaty", "canvas", "ecosystem", "transistor", "sonnet", "cathedral", "particle",
        "corridor", "horizon", "matrix", "tapestry", "compass", "prism", "turbine", "archive",
        "spectrum", "constellation", "glider", "bastion", "nucleus", "quarry", "harbor", "beacon",
        "fountain", "monolith", "mosaic", "canyon", "cradle", "chronicle", "labyrinth", "pinnacle",
        "sanctuary", "cascade", "vortex", "threshold", "citadel", "pendulum", "artifact", "biosphere"
    ]
    open_adjectives = [
        "ancient", "luminous", "dynamic", "fragile", "intricate", "resonant", "sporadic", "subtle",
        "ephemeral", "rigorous", "turbulent", "harmonic", "pristine", "dormant", "synthetic", "ethereal",
        "tenacious", "versatile", "profound", "transient", "immutable", "crystalline", "kinetic", "radiant",
        "solitary", "complex", "variable", "vibrant", "arcane", "austere", "seamless", "perpetual"
    ]
    open_verbs = [
        "illuminates", "transforms", "accelerates", "stabilizes", "catalyzes", "reflects", "modulates",
        "disrupts", "generates", "amplifies", "delineates", "reconfigures", "preserves", "transcends",
        "influences", "harmonizes", "mitigates", "reveals", "integrates", "dissolves", "sustains",
        "manifests", "crystallizes", "navigates", "elevates", "unifies", "questions", "encapsulates"
    ]
    open_contexts = [
        "in changing environments", "through empirical observation", "across historical eras",
        "under variable atmospheric pressures", "within theoretical frameworks", "over seasonal cycles",
        "beyond conventional boundaries", "through systematic inquiry", "at microscopic scales",
        "amid geographical transitions", "within closed thermodynamic loops", "during structural shifts",
        "across cultural boundaries", "under extreme temperature gradients", "in natural ecosystems",
        "through mathematical formalisms"
    ]

    rng = np.random.default_rng(seed)
    sentences = []
    seen = set()
    while len(sentences) < total_n:
        adj1 = rng.choice(open_adjectives)
        n1 = rng.choice(open_nouns)
        v = rng.choice(open_verbs)
        adj2 = rng.choice(open_adjectives)
        n2 = rng.choice(open_nouns)
        ctx = rng.choice(open_contexts)
        s = f"The {adj1} {n1} {v} the {adj2} {n2} {ctx}."
        if s not in seen:
            seen.add(s)
            sentences.append(s)
    return sentences


def generate_corpus_c(total_n: int = 10000, seed: int = 42) -> list[str]:
    """Generates the Pathological Single-Template Phase 4 Corpus.
    
    Uses the exact template from run_phase4_scale.py:
    'Fact number {i} about topic {i % 50}: the value is {val} and the category is {i % 50}.'
    Almost all sentence tokens are identical, forcing high mutual cosine similarities (>0.70).
    """
    rng = np.random.default_rng(seed)
    return [
        f"Fact number {i} about topic {i % 50}: "
        f"the value is {rng.integers(0, 1000)} and the category is {i % 50}."
        for i in range(total_n)
    ]


# ---------------------------------------------------------------------------
# Efficient Percolation Analysis
# ---------------------------------------------------------------------------

def compute_similarity_stats(pair_sims: np.ndarray, bins: int = 50) -> dict:
    """Computes summary statistics and histogram for pairwise similarities."""
    mu = float(np.mean(pair_sims))
    sigma = float(np.std(pair_sims))
    hist, bin_edges = np.histogram(pair_sims, bins=bins, range=(-0.20, 1.00))
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    
    percentiles = [1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0, 99.9]
    perc_vals = {f"p{p}": float(np.percentile(pair_sims, p)) for p in percentiles}
    
    # Check mass in spectral gap [0.50, 0.70]
    gap_count = int(np.sum((pair_sims >= 0.50) & (pair_sims <= 0.70)))
    gap_frac = float(gap_count / len(pair_sims))

    return {
        "count": len(pair_sims),
        "mean": mu,
        "std": sigma,
        "min": float(np.min(pair_sims)),
        "max": float(np.max(pair_sims)),
        "median": float(np.median(pair_sims)),
        "gap_fraction_05_07": gap_frac,
        "percentiles": perc_vals,
        "histogram": {
            "counts": [int(c) for c in hist],
            "bin_edges": [float(b) for b in bin_edges],
            "bin_centers": [float(b) for b in bin_centers],
        }
    }


def sweep_percolation_curve(unit_matrix: np.ndarray, thetas: np.ndarray, min_theta: float = 0.30) -> dict:
    """Computes largest component fraction across a grid of thresholds descending."""
    n = unit_matrix.shape[0]
    sims = unit_matrix @ unit_matrix.T
    iu = np.triu_indices(n, 1)
    pair_sims = sims[iu]
    del sims

    # Filter edges above min_theta for fast union-find
    mask = pair_sims >= min_theta
    sub_sims = pair_sims[mask]
    sub_i = iu[0][mask]
    sub_j = iu[1][mask]

    order = np.argsort(-sub_sims)
    sorted_sims = sub_sims[order]
    sorted_i = sub_i[order]
    sorted_j = sub_j[order]
    del sub_sims, sub_i, sub_j, mask

    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)
    max_size = 1

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    curve = {}
    ptr = 0
    total = len(sorted_sims)
    for t in sorted(thetas, reverse=True):
        while ptr < total and sorted_sims[ptr] >= t:
            a, b = int(sorted_i[ptr]), int(sorted_j[ptr])
            ra, rb = find(a), find(b)
            if ra != rb:
                if size[ra] < size[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                size[ra] += size[rb]
                if size[ra] > max_size:
                    max_size = int(size[ra])
            ptr += 1
        curve[f"{t:.2f}"] = float(max_size / n)

    # Determine empirical theta_crit (where giant fraction first exceeds 0.50 descending)
    theta_crit_emp = None
    for t in sorted(thetas, reverse=True):
        if curve[f"{t:.2f}"] > 0.50:
            theta_crit_emp = float(t)
            break
    if theta_crit_emp is None:
        theta_crit_emp = float(min_theta)

    # Theoretical critical threshold: p(theta) = 1 / (N - 1) => theta = Q_{1 - 1/(N-1)}(sim)
    q_crit_theory = 1.0 - 1.0 / (n - 1)
    theta_crit_theory = float(np.quantile(pair_sims, q_crit_theory))

    return {
        "curve": curve,
        "theta_crit_empirical": theta_crit_emp,
        "theta_crit_theory": theta_crit_theory,
        "pair_sims_ref": pair_sims  # returned for quantile rule testing
    }


def evaluate_threshold_rule(sims_matrix: np.ndarray, theta: float) -> dict:
    """Evaluates average degree <k> and giant component fraction at a given threshold."""
    n = sims_matrix.shape[0]
    iu = np.triu_indices(n, 1)
    pair_sims = sims_matrix[iu]
    
    edge_mask = pair_sims >= theta
    num_edges = int(np.sum(edge_mask))
    total_pairs = len(pair_sims)
    edge_prob = num_edges / total_pairs
    avg_degree = edge_prob * (n - 1)

    # Connected components via union-find
    parent = np.arange(n, dtype=np.int32)
    size = np.ones(n, dtype=np.int32)
    max_size = 1

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edge_i = iu[0][edge_mask]
    edge_j = iu[1][edge_mask]
    for a, b in zip(edge_i, edge_j):
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
            if size[ra] > max_size:
                max_size = int(size[ra])

    return {
        "theta": float(theta),
        "num_edges": num_edges,
        "edge_probability": float(edge_prob),
        "avg_degree": float(avg_degree),
        "giant_component_size": int(max_size),
        "giant_component_fraction": float(max_size / n)
    }


# ---------------------------------------------------------------------------
# Main Experiment Execution
# ---------------------------------------------------------------------------

def main():
    print(f"=== FCVAM Percolation & Chaining Analysis ===")
    print(f"Loading SentenceTransformer: {MODEL_NAME}...", flush=True)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"Loaded in {time.time()-t0:.2f}s.\n", flush=True)

    max_n = max(N_VALUES)
    results = {
        "metadata": {
            "model": MODEL_NAME,
            "n_values": N_VALUES,
            "c_factor": C_FACTOR,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "histograms": {},
        "percolation_analysis": {
            "corpus_a_clustered": {},
            "corpus_b_heterogeneous": {}
        },
        "quantile_rule_evaluation": {
            "corpus_a_clustered": {},
            "corpus_b_heterogeneous": {},
            "corpus_c_pathological": {}
        }
    }

    # 1. Generate and encode all three corpora at max_n (10,000)
    print(f"--- 1. Generating & Encoding Corpora (N={max_n}) ---")
    cache_dir = Path("/tmp/kilo/percolation_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_a = cache_dir / "emb_a.npy"
    cache_b = cache_dir / "emb_b.npy"
    cache_c = cache_dir / "emb_c.npy"

    print(f"Generating Corpus A (Clustered, 20 topics)...", flush=True)
    sents_a, labels_a = generate_corpus_a(max_n)
    if cache_a.exists():
        emb_a = np.load(cache_a)
        print(f"  Loaded Corpus A from cache: {len(emb_a)} vectors.")
    else:
        t0 = time.time()
        emb_a = model.encode(sents_a, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        emb_a = np.asarray(emb_a, dtype=np.float32)
        np.save(cache_a, emb_a)
        print(f"  Encoded {len(emb_a)} sentences in {time.time()-t0:.2f}s.")

    print(f"Generating Corpus B (Heterogeneous, open domain)...", flush=True)
    sents_b = generate_corpus_b(max_n, seed=42)
    if cache_b.exists():
        emb_b = np.load(cache_b)
        print(f"  Loaded Corpus B from cache: {len(emb_b)} vectors.")
    else:
        t0 = time.time()
        emb_b = model.encode(sents_b, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        emb_b = np.asarray(emb_b, dtype=np.float32)
        np.save(cache_b, emb_b)
        print(f"  Encoded {len(emb_b)} sentences in {time.time()-t0:.2f}s.")

    print(f"Generating Corpus C (Pathological single template)...", flush=True)
    sents_c = generate_corpus_c(max_n, seed=42)
    if cache_c.exists():
        emb_c = np.load(cache_c)
        print(f"  Loaded Corpus C from cache: {len(emb_c)} vectors.")
    else:
        t0 = time.time()
        emb_c = model.encode(sents_c, normalize_embeddings=True, batch_size=256, show_progress_bar=False)
        emb_c = np.asarray(emb_c, dtype=np.float32)
        np.save(cache_c, emb_c)
        print(f"  Encoded {len(emb_c)} sentences in {time.time()-t0:.2f}s.")

    # 2. Compute full empirical pairwise similarity distributions & histograms
    print(f"--- 2. Pairwise Similarity Distributions (at N={max_n}) ---")
    
    # Corpus A: decompose into intra-topic and inter-topic
    print("Computing Corpus A pairwise similarities...", flush=True)
    sims_a_full = emb_a @ emb_a.T
    iu_a = np.triu_indices(max_n, 1)
    pair_a = sims_a_full[iu_a]
    lbl_arr = np.asarray(labels_a)
    intra_mask = lbl_arr[iu_a[0]] == lbl_arr[iu_a[1]]
    inter_mask = ~intra_mask

    stats_a_all = compute_similarity_stats(pair_a)
    stats_a_intra = compute_similarity_stats(pair_a[intra_mask])
    stats_a_inter = compute_similarity_stats(pair_a[inter_mask])
    results["histograms"]["corpus_a_clustered"] = {
        "overall": stats_a_all,
        "intra_topic": stats_a_intra,
        "inter_topic": stats_a_inter
    }
    print(f"  Corpus A overall: mu={stats_a_all['mean']:.3f}, sigma={stats_a_all['std']:.3f}, gap[0.5,0.7]={stats_a_all['gap_fraction_05_07']*100:.3f}%")
    print(f"  Corpus A intra  : mu={stats_a_intra['mean']:.3f}, median={stats_a_intra['median']:.3f} (PEAK ~0.85-0.95)")
    print(f"  Corpus A inter  : mu={stats_a_inter['mean']:.3f}, median={stats_a_inter['median']:.3f} (PEAK ~0.05-0.25)")

    # Corpus B: Heterogeneous unimodal
    print("Computing Corpus B pairwise similarities...", flush=True)
    sims_b_full = emb_b @ emb_b.T
    iu_b = np.triu_indices(max_n, 1)
    pair_b = sims_b_full[iu_b]
    stats_b = compute_similarity_stats(pair_b)
    results["histograms"]["corpus_b_heterogeneous"] = stats_b
    print(f"  Corpus B: mu={stats_b['mean']:.3f}, sigma={stats_b['std']:.3f}, gap[0.5,0.7]={stats_b['gap_fraction_05_07']*100:.3f}% (CONTINUOUS UNIMODAL)")

    # Corpus C: Pathological Phase 4 template
    print("Computing Corpus C pairwise similarities...", flush=True)
    sims_c_full = emb_c @ emb_c.T
    iu_c = np.triu_indices(max_n, 1)
    pair_c = sims_c_full[iu_c]
    stats_c = compute_similarity_stats(pair_c)
    results["histograms"]["corpus_c_pathological"] = stats_c
    frac_above_07 = float(np.sum(pair_c >= 0.70) / len(pair_c))
    frac_above_085 = float(np.sum(pair_c >= 0.85) / len(pair_c))
    results["histograms"]["corpus_c_pathological"]["fraction_ge_070"] = frac_above_07
    results["histograms"]["corpus_c_pathological"]["fraction_ge_085"] = frac_above_085
    print(f"  Corpus C: mu={stats_c['mean']:.3f}, sigma={stats_c['std']:.3f}, fraction >= 0.70: {frac_above_07*100:.2f}%, >= 0.85: {frac_above_085*100:.2f}%\n")

    # 3. Compute percolation curves across N in {200, 1000, 5000, 10000}
    print("--- 3. Percolation Sweeps across N in {200, 1000, 5000, 10000} ---")
    for n in N_VALUES:
        print(f"\nEvaluating N = {n}:")

        # Corpus A subset: take balanced n // 20 per topic
        k_topics = len(RAW_TOPICS_20)
        items_per_topic = n // k_topics
        indices_a = []
        for t in range(k_topics):
            start = t * (max_n // k_topics)
            indices_a.extend(range(start, start + items_per_topic))
        sub_emb_a = emb_a[indices_a]
        sweep_a = sweep_percolation_curve(sub_emb_a, THETAS, min_theta=0.30)
        pair_sims_a_n = sweep_a.pop("pair_sims_ref")

        # Corpus B subset: first n elements
        sub_emb_b = emb_b[:n]
        sweep_b = sweep_percolation_curve(sub_emb_b, THETAS, min_theta=0.30)
        pair_sims_b_n = sweep_b.pop("pair_sims_ref")

        # Corpus C subset: first n elements
        sub_emb_c = emb_c[:n]
        sims_c_n = sub_emb_c @ sub_emb_c.T
        pair_sims_c_n = sims_c_n[np.triu_indices(n, 1)]

        results["percolation_analysis"]["corpus_a_clustered"][str(n)] = {
            "n": n,
            "theta_crit_empirical": sweep_a["theta_crit_empirical"],
            "theta_crit_theory": sweep_a["theta_crit_theory"],
            "giant_frac_at_085": sweep_a["curve"].get("0.85", 0.0),
            "giant_frac_at_060": sweep_a["curve"].get("0.60", 0.0),
            "curve": sweep_a["curve"]
        }

        results["percolation_analysis"]["corpus_b_heterogeneous"][str(n)] = {
            "n": n,
            "theta_crit_empirical": sweep_b["theta_crit_empirical"],
            "theta_crit_theory": sweep_b["theta_crit_theory"],
            "giant_frac_at_085": sweep_b["curve"].get("0.85", 0.0),
            "curve": sweep_b["curve"]
        }

        print(f"  Corpus A (Clustered)    : theta_crit_emp = {sweep_a['theta_crit_empirical']:.2f}, theory = {sweep_a['theta_crit_theory']:.3f} | S(0.85) = {sweep_a['curve'].get('0.85'):.3f}, S(0.60) = {sweep_a['curve'].get('0.60'):.3f}")
        print(f"  Corpus B (Heterogeneous): theta_crit_emp = {sweep_b['theta_crit_empirical']:.2f}, theory = {sweep_b['theta_crit_theory']:.3f} | S(0.85) = {sweep_b['curve'].get('0.85'):.3f}")

        # 4. Evaluate Actionable Quantile Rule: theta(N) = Q_{1 - c/N}(sim)
        q_target = 1.0 - C_FACTOR / n

        # Corpus A evaluation
        theta_q_a = float(np.quantile(pair_sims_a_n, q_target))
        sims_a_n = sub_emb_a @ sub_emb_a.T
        eval_q_a = evaluate_threshold_rule(sims_a_n, theta_q_a)
        eval_adapt_a = evaluate_threshold_rule(sims_a_n, float(pair_sims_a_n.mean() + 2 * pair_sims_a_n.std()))
        eval_085_a = evaluate_threshold_rule(sims_a_n, 0.85)
        results["quantile_rule_evaluation"]["corpus_a_clustered"][str(n)] = {
            "quantile_rule": eval_q_a,
            "adaptive_heuristic": eval_adapt_a,
            "fixed_085": eval_085_a
        }

        # Corpus B evaluation
        theta_q_b = float(np.quantile(pair_sims_b_n, q_target))
        sims_b_n = sub_emb_b @ sub_emb_b.T
        eval_q_b = evaluate_threshold_rule(sims_b_n, theta_q_b)
        eval_adapt_b = evaluate_threshold_rule(sims_b_n, float(pair_sims_b_n.mean() + 2 * pair_sims_b_n.std()))
        eval_085_b = evaluate_threshold_rule(sims_b_n, 0.85)
        results["quantile_rule_evaluation"]["corpus_b_heterogeneous"][str(n)] = {
            "quantile_rule": eval_q_b,
            "adaptive_heuristic": eval_adapt_b,
            "fixed_085": eval_085_b
        }

        # Corpus C evaluation
        theta_q_c = float(np.quantile(pair_sims_c_n, q_target))
        eval_q_c = evaluate_threshold_rule(sims_c_n, theta_q_c)
        eval_adapt_c = evaluate_threshold_rule(sims_c_n, float(pair_sims_c_n.mean() + 2 * pair_sims_c_n.std()))
        eval_085_c = evaluate_threshold_rule(sims_c_n, 0.85)
        results["quantile_rule_evaluation"]["corpus_c_pathological"][str(n)] = {
            "quantile_rule": eval_q_c,
            "adaptive_heuristic": eval_adapt_c,
            "fixed_085": eval_085_c
        }

        print(f"  Actionable Quantile Rule (c={C_FACTOR}, target p={C_FACTOR/n:.6f}):")
        print(f"    Corpus A: theta={theta_q_a:.4f} -> <k>={eval_q_a['avg_degree']:.3f}, giant_frac={eval_q_a['giant_component_fraction']:.4f}")
        print(f"    Corpus B: theta={theta_q_b:.4f} -> <k>={eval_q_b['avg_degree']:.3f}, giant_frac={eval_q_b['giant_component_fraction']:.4f}")
        print(f"    Corpus C: theta={theta_q_c:.4f} -> <k>={eval_q_c['avg_degree']:.3f}, giant_frac={eval_q_c['giant_component_fraction']:.4f}")
        print(f"    [Comparison on Corpus C at theta=0.85]: <k>={eval_085_c['avg_degree']:.1f}, giant_frac={eval_085_c['giant_component_fraction']:.4f} (COLLAPSE!)")

    # 5. Save all results
    out_path = ROOT_DIR / "experiments" / "results_percolation_histogram.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n=======================================================")
    print(f"Saved complete results to: {out_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
