#!/usr/bin/env python3
"""nhic_randomised_pipeline.py

A runnable reference implementation for **Case Study A – Randomised Algorithms for Sensor Network
Anomaly Detection**.

This script demonstrates a streaming-capable anomaly detection pipeline using:
  1) Weighted Reservoir Sampling (to keep a small, biased sample of important events)
  2) Count-Min Sketch (to track approximate anomaly frequency / "mass" with fixed memory)
  3) Locality-Sensitive Hashing (LSH) via random hyperplanes (to bucket similar anomaly events)
  4) Probabilistic imputation for missing/corrupted readings (returns a distribution, not a mean)

Why "sliding time window per region"?
-------------------------------------
The assignment brief requires streaming algorithms and sublinear memory.
We therefore do NOT keep all historical readings.

Instead, for each region we keep a bounded buffer of only the most recent readings
within a fixed time window (e.g., last 120 seconds), capped to a maximum number of
stored items. This is a classic "sliding window" strategy and aligns with:
  - Spatial similarity (region-based context)
  - Streaming throughput (bounded memory)
  - Real-time imputation (use only recent local context)

Run:
-----
  python nhic_randomised_pipeline.py

You should see alerts printed for:
  - imputation events (missing/corrupted)
  - suspected spoofed spikes
  - cluster anomalies (LSH bucket collisions)

Note on "minute windows":
-------------------------
In the report, S204 is imputed using other *same minute* readings from region R3.
If we process records strictly one-by-one in arrival order, you cannot use future
readings that have not arrived yet.

To match the report and the assignment's "one reading per minute" framing, this
implementation provides a `process_minute_window(...)` helper that processes a
small 1-minute micro-batch in two passes:
  (1) ingest observed readings for context, then
  (2) impute missing/corrupt readings using that same-minute context.

This is still streaming compatible (windowed streaming is standard in Flink/Spark).

This file is deliberately verbose with inline comments for report/appendix use.
"""

from __future__ import annotations

import math
import random
import time
import hashlib
import heapq
from dataclasses import dataclass
from collections import deque, defaultdict
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


# -----------------------------
# 0) Small utilities
# -----------------------------

def stable_hash_int(key: Any, seed: int, mod: int) -> int:
    """Return a *stable* hash integer in [0, mod).

    Why stable hashing?
    - Python's built-in hash() is salted per process, so values change each run.
    - For deterministic behaviour in examples/tests, we use a cryptographic hash.

    This is not meant to be cryptographically secure here; it is meant to be stable.
    """
    h = hashlib.blake2b(digest_size=8)
    payload = f"{seed}|{repr(key)}".encode("utf-8")
    h.update(payload)
    val = int.from_bytes(h.digest(), byteorder="little", signed=False)
    return val % mod


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x into [lo, hi]."""
    return max(lo, min(hi, x))


# -----------------------------
# 1) Data model: a single reading from the stream
# -----------------------------

@dataclass(frozen=True)
class Reading:
    """One sensor record (one minute / one event).

    value:
      - float: valid measurement
      - None: missing or corrupted reading (the assignment uses '?')

    conf:
      - signal confidence in [0, 1]
    """

    sensor_id: str
    region: str
    value: Optional[float]
    conf: float
    t: float  # unix timestamp (seconds)


# -----------------------------
# 2) Count-Min Sketch (CMS)
# -----------------------------

class CountMinSketch:
    """Count-Min Sketch for approximate frequency / aggregate tracking.

    - Fixed memory: width * depth counters
    - Streaming: update is O(depth) per item
    - Query gives an *over estimate* with controllable error probability

    Here we use float counters so we can track:
      - anomaly_count : count of anomaly events
      - anomaly_mass  : sum of anomaly scores (a "severity" proxy)
    """

    def __init__(self, width: int = 4000, depth: int = 5, seed: int = 1337):
        self.width = int(width)
        self.depth = int(depth)
        self.seeds = [seed + i * 10007 for i in range(self.depth)]
        self.table: List[List[float]] = [[0.0] * self.width for _ in range(self.depth)]

    def update(self, key: Any, increment: float = 1.0) -> None:
        """Update CMS counters for `key` by `increment`."""
        for row, s in enumerate(self.seeds):
            col = stable_hash_int(key, s, self.width)
            self.table[row][col] += increment

    def query(self, key: Any) -> float:
        """Return approximate count (min across rows)."""
        est = float("inf")
        for row, s in enumerate(self.seeds):
            col = stable_hash_int(key, s, self.width)
            est = min(est, self.table[row][col])
        return est


# -----------------------------
# 3) Weighted Reservoir Sampling (A-Res style)
# -----------------------------

class WeightedReservoirSampler:
    """Keep a fixed size reservoir of the *most important* stream items.

    Reservoir sampling is essential for a large sensor stream because:
      - 1.2M sensors/min is too much to store
      - yet we want some raw examples for audit/forensics

    Here we implement a standard weighted scheme:
      - Each item gets a random key derived from its weight
      - Keep the top-k keys in a min-heap

    Key idea:
      - larger weight => more likely to be kept
      - still randomized => avoids deterministic bias
    """

    def __init__(self, k: int, rng: Optional[random.Random] = None):
        self.k = int(k)
        self.rng = rng or random.Random(42)
        # Heap contains (priority_key, item). Min-heap so smallest key is at index 0.
        self._heap: List[Tuple[float, Reading]] = []

    def update(self, item: Reading, weight: float) -> None:
        """Try to insert `item` into the reservoir using `weight`."""
        if weight <= 0.0:
            return

        # A-Res key: u^(1/weight), with u ~ Uniform(0,1).
        # Larger weight increases chance of having a larger key.
        u = self.rng.random()
        key = u ** (1.0 / weight)

        if len(self._heap) < self.k:
            heapq.heappush(self._heap, (key, item))
            return

        # If this item outranks the current smallest, replace it.
        if key > self._heap[0][0]:
            heapq.heapreplace(self._heap, (key, item))

    def items(self) -> List[Reading]:
        """Return reservoir items in descending key order (most important first)."""
        return [it for _, it in sorted(self._heap, reverse=True)]


# -----------------------------
# 4) LSH: Random Hyperplane signatures for approximate similarity
# -----------------------------

class RandomHyperplaneLSH:
    """Locality Sensitive Hashing using random hyperplanes.

    We map a numeric feature vector x into a bit-signature:
      bit_j = 1 if dot(r_j, x) >= 0 else 0

    Similar vectors tend to share similar signatures.

    We then use banding to create multiple bucket keys per signature.
    If multiple anomaly events land in the same bucket within a time window,
    we treat that as a candidate anomalous cluster.
    """

    def __init__(self, dim: int, num_bits: int = 32, band_size: int = 8, seed: int = 123):
        if num_bits % band_size != 0:
            raise ValueError("num_bits must be divisible by band_size")

        self.dim = int(dim)
        self.num_bits = int(num_bits)
        self.band_size = int(band_size)
        self.num_bands = self.num_bits // self.band_size

        self.rng = random.Random(seed)

        # Random hyperplanes: list of num_bits vectors of length `dim`
        self.hyperplanes: List[List[float]] = [
            [self._randn() for _ in range(self.dim)] for _ in range(self.num_bits)
        ]

    def _randn(self) -> float:
        """Generate a pseudo-normal variate (Box-Muller)."""
        u1 = max(1e-12, self.rng.random())
        u2 = self.rng.random()
        return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)

    def signature(self, x: List[float]) -> int:
        """Return the packed integer bit-signature for feature vector x."""
        if len(x) != self.dim:
            raise ValueError(f"Expected dim={self.dim}, got len(x)={len(x)}")

        sig = 0
        for j, hp in enumerate(self.hyperplanes):
            dot = 0.0
            for i in range(self.dim):
                dot += hp[i] * x[i]
            if dot >= 0.0:
                sig |= (1 << j)
        return sig

    def band_buckets(self, sig: int) -> List[str]:
        """Split signature into bands and return list of bucket keys."""
        buckets: List[str] = []
        mask = (1 << self.band_size) - 1

        for b in range(self.num_bands):
            band_bits = (sig >> (b * self.band_size)) & mask
            # Bucket key includes band index to avoid mixing bands.
            buckets.append(f"band{b}:{band_bits}")

        return buckets


# -----------------------------
# 5) Sliding-window structures for clustering and region context
# -----------------------------

@dataclass
class BucketEvent:
    """Event stored inside an LSH bucket for cluster detection."""

    sensor_id: str
    region: str
    value: float
    conf: float
    score: float
    t: float


class SlidingBucket:
    """Holds events in a bucket for a fixed time window.

    - We keep only the events from the last `window_seconds`.
    - This supports "cluster in real-time" detection.

    This is a streaming analogue of clustering within a short time window.
    """

    def __init__(self, window_seconds: float = 60.0):
        self.window_seconds = float(window_seconds)
        self.events: Deque[BucketEvent] = deque()
        self.sum_score: float = 0.0

    def add(self, ev: BucketEvent) -> None:
        self.events.append(ev)
        self.sum_score += ev.score

    def expire(self, now_t: float) -> None:
        """Drop events older than the bucket's time window."""
        while self.events and (now_t - self.events[0].t) > self.window_seconds:
            old = self.events.popleft()
            self.sum_score -= old.score

    @property
    def count(self) -> int:
        return len(self.events)

    def members(self) -> List[str]:
        return [e.sensor_id for e in self.events]


@dataclass
class RegionObs:
    """One observation stored for region context (imputation + optional features)."""

    value: float
    conf: float
    t: float


class RegionBuffer:
    """A bounded, sliding-time-window buffer of recent readings for a region.

    Important:
    - This is NOT meant to store all sensors.
    - We store a limited number of *recent* observations (maxlen) in a time window.

    This aligns with the assignment's streaming constraint:
      - limited memory (sublinear)
      - recent data is most relevant for local imputation
    """

    def __init__(self, window_seconds: float = 120.0, maxlen: int = 500):
        self.window_seconds = float(window_seconds)
        self.maxlen = int(maxlen)
        self._dq: Deque[RegionObs] = deque()

    def add(self, value: float, conf: float, t: float) -> None:
        """Add a new reading and expire old entries."""
        self._dq.append(RegionObs(value=value, conf=conf, t=t))
        self.expire(t)

        # Enforce maxlen cap as a hard memory bound.
        while len(self._dq) > self.maxlen:
            self._dq.popleft()

    def expire(self, now_t: float) -> None:
        """Remove readings older than the region time window."""
        while self._dq and (now_t - self._dq[0].t) > self.window_seconds:
            self._dq.popleft()

    def recent(self, now_t: float) -> List[RegionObs]:
        """Return the current contents (already time-filtered)."""
        self.expire(now_t)
        return list(self._dq)


# -----------------------------
# 6) Domain logic: anomaly scoring, sampling weights, features, and imputation
# -----------------------------

HIGH_RISK_REGIONS = {"R2", "R3"}


def confidence_weighting_function(conf: float, gamma: float = 2.0) -> float:
    """The *confidence weighting function* f(conf).

    In the report you can name this explicitly as:
      - confidence weighting function

    We use f(conf) = conf^gamma, with gamma>1 to aggressively downweight low-confidence spikes.

    This matches the brief requirement:
      - Readings >145 are potential anomalies but must be weighted by signal confidence.
    """
    conf = clamp(conf, 0.0, 1.0)
    return conf ** float(gamma)


def anomaly_score(value: float, conf: float, region: str, threshold: float = 145.0) -> float:
    """Compute a confidence weighted anomaly score.

    score = region_risk_multiplier * max(0, value - threshold) * f(conf)

    - max(0, value threshold) ensures no contribution below threshold.
    - f(conf) suppresses low confidence spikes (spoofing/noise).
    - region multiplier makes R2/R3 more significant.
    """
    risk = 1.35 if region in HIGH_RISK_REGIONS else 1.0
    exceed = max(0.0, value - threshold)
    return risk * exceed * confidence_weighting_function(conf, gamma=2.0)


def sampling_weight(score: float, conf: float, region: str) -> float:
    """Weight used by the reservoir sampler.

    Design intent:
      - higher score => more likely to be kept
      - high-risk region => slightly more likely to be kept
      - confidence contributes but cannot dominate by itself
    """
    risk = 1.20 if region in HIGH_RISK_REGIONS else 1.0
    return risk * (0.5 + clamp(conf, 0.0, 1.0)) * (1.0 + score)


def make_features(value: float, conf: float, region: str) -> List[float]:
    """Feature vector for LSH.

    Keep features simple, stable and fast:
      - normalised reading magnitude
      - confidence
      - region code

    In a real system, you may add rolling mean/std or trend features.
    Here we keep it minimal to align with the brief and avoid heavy per event computation.
    """
    region_code = {"R1": 0.0, "R2": 1.0, "R3": 2.0, "R4": 3.0}.get(region, 0.0)

    # Normalize value so ranges are comparable in dot products.
    # (This is a heuristic normalisation for demo purposes.)
    value_norm = (value - 100.0) / 50.0

    return [
        value_norm,
        clamp(conf, 0.0, 1.0),
        region_code / 3.0,
    ]


def impute_distribution_from_region(
    region_obs: List[RegionObs],
    conf_missing: float,
    now_t: float,
    tau_seconds: float = 60.0,
    base_noise: float = 2.0,
    # NOTE: extra_noise controls how much uncertainty we add when confidence is low.
    # A moderate default (30) keeps high-confidence imputations relatively tight
    # while still making low-confidence imputations wide.
    extra_noise: float = 30.0,
    min_neighbor_conf: float = 0.20,
) -> Tuple[float, float]:
    """Probabilistic reconstruction of a missing/corrupted reading.

    Returns (mu, sigma), representing a Normal(mu, sigma^2) predictive distribution.

    How it matches the brief:
      - Spatial similarity: uses only region-level context (region_obs)
      - Confidence weights: neighbors contribute by their conf
      - Noise model: sigma increases when conf_missing is low

    Weighting scheme:
      - confidence-weighted AND time-decayed:
            w_i = f(conf_i) * exp(-(now_t - t_i)/tau)

        where f(conf) is the confidence weighting function (we use conf^2).

      - optionally ignore extremely low-confidence neighbors (min_neighbor_conf)
        to reduce the influence of suspected spoof/noise readings.

    Final uncertainty:
      - neighbor variability + sensor noise
      - sensor noise increases when conf_missing is low
    """
    conf_missing = clamp(conf_missing, 0.0, 1.0)

    if not region_obs:
        # If we have no context, fall back to a broad prior.
        # In a real deployment, this could be region-specific.
        return 100.0, 50.0

    # Filter out very low-confidence neighbors to reduce spoof impact.
    filtered = [obs for obs in region_obs if clamp(obs.conf, 0.0, 1.0) >= min_neighbor_conf]

    if not filtered:
        # If everything is low-confidence, we cannot trust the local context.
        return 100.0, 50.0

    weighted: List[Tuple[float, float]] = []
    for obs in filtered:
        dt = max(0.0, now_t - obs.t)
        # Confidence-weighted (conf^2) and time-decayed contribution
        w = confidence_weighting_function(obs.conf, gamma=2.0) * math.exp(-dt / max(1e-6, tau_seconds))
        weighted.append((obs.value, w))

    w_sum = sum(w for _, w in weighted)
    if w_sum <= 1e-12:
        # All weights vanished (can happen with extreme decay). Use broad prior.
        return 100.0, 50.0

    mu = sum(v * w for v, w in weighted) / w_sum

    # Weighted variance of neighbors (how consistent they are)
    var_neighbors = sum(w * (v - mu) ** 2 for v, w in weighted) / w_sum

    # Sensor-specific noise based on the missing sensor's confidence.
    # Lower confidence => more uncertainty.
    sigma_sensor = base_noise + (1.0 - conf_missing) * extra_noise

    sigma = math.sqrt(var_neighbors + sigma_sensor ** 2)
    return mu, sigma


# -----------------------------
# 7) Main pipeline class
# -----------------------------

class NHICPipeline:
    """Randomised streaming anomaly detection pipeline.

    Components:
      - WeightedReservoirSampler: keeps a small sample of important events
      - CountMinSketch: tracks approximate anomaly stats per region
      - RandomHyperplaneLSH + SlidingBucket: detects clusters of similar anomalies
      - RegionBuffer: provides sliding-window region context for probabilistic imputation

    Output:
      - process(reading) returns a list of alert dictionaries
    """

    def __init__(
        self,
        # Reservoir sampling
        reservoir_k: int = 2000,
        # CMS parameters
        cms_width: int = 4000,
        cms_depth: int = 5,
        # LSH parameters
        lsh_bits: int = 32,
        lsh_band_size: int = 8,
        # Sliding windows
        bucket_window_seconds: float = 60.0,
        region_window_seconds: float = 120.0,
        region_buffer_maxlen: int = 500,
        # Decision thresholds
        score_min: float = 1.0,
        min_cluster_size: int = 3,
        min_cluster_score: float = 5.0,
        # Spoofing heuristic threshold
        spoof_conf_max: float = 0.20,
    ):
        # Random sampling for "inspectable" events
        self.reservoir = WeightedReservoirSampler(k=reservoir_k)

        # Sketch for approximate anomaly counters
        self.cms = CountMinSketch(width=cms_width, depth=cms_depth)

        # LSH for similarity bucketing
        self.lsh = RandomHyperplaneLSH(dim=3, num_bits=lsh_bits, band_size=lsh_band_size)

        # LSH buckets (each bucket has its own sliding time window)
        self.buckets: Dict[str, SlidingBucket] = defaultdict(lambda: SlidingBucket(bucket_window_seconds))

        # Region-level sliding context (bounded memory per region)
        self.region_buffers: Dict[str, RegionBuffer] = defaultdict(
            lambda: RegionBuffer(window_seconds=region_window_seconds, maxlen=region_buffer_maxlen)
        )

        # Decision thresholds
        self.score_min = float(score_min)
        self.min_cluster_size = int(min_cluster_size)
        self.min_cluster_score = float(min_cluster_score)

        # Spoofing threshold: low-confidence spikes are suspicious
        self.spoof_conf_max = float(spoof_conf_max)

    # -------------------------
    # Internal helper (shared by both online + windowed processing)
    # -------------------------
    def _process_observed(
        self,
        r: Reading,
        *,
        value: float,
        conf: float,
        update_region_buffer: bool,
    ) -> List[Dict[str, Any]]:
        """Process a *non-missing* reading.

        Parameters
        ----------
        update_region_buffer:
            - True  => behave like fully-online streaming (append to region buffer now)
            - False => region buffer already contains this window's readings
                      (used by process_minute_window)
        """
        alerts: List[Dict[str, Any]] = []

        # (Optional) Update region context (sliding window per region).
        # In windowed mode we do this in a separate pass.
        if update_region_buffer:
            self.region_buffers[r.region].add(value=value, conf=conf, t=r.t)

        # -------------------------
        # Spoof suspicion heuristic (must happen even if anomaly score is low)
        # -------------------------
        # The prompt explicitly highlights S203 as "likely spoofed".
        # That sensor would have a low anomaly score due to low confidence,
        # but we still want to surface it as a spoofing attempt.
        if value > 145.0 and conf < self.spoof_conf_max:
            alerts.append(
                {
                    "type": "spoof_suspect",
                    "sensor": r.sensor_id,
                    "region": r.region,
                    "value": value,
                    "conf": conf,
                    "note": "High reading with very low confidence (possible spoofing)",
                }
            )

        # -------------------------
        # Compute anomaly score (confidence-weighted + region risk)
        # -------------------------
        score = anomaly_score(value=value, conf=conf, region=r.region)

        # If score is too small, skip heavier processing (streaming efficiency).
        # NOTE: we still returned spoof alerts above if relevant.
        if score < self.score_min:
            return alerts

        # -------------------------
        # Reservoir sampling: keep important events for inspection
        # -------------------------
        w = sampling_weight(score=score, conf=conf, region=r.region)
        self.reservoir.update(item=r, weight=w)

        # -------------------------
        # Sketching: update approximate anomaly indicators
        # -------------------------
        self.cms.update((r.region, "anomaly_count"), 1.0)
        self.cms.update((r.region, "anomaly_mass"), score)

        # -------------------------
        # LSH bucketing: detect clustered anomalies
        # -------------------------
        x = make_features(value=value, conf=conf, region=r.region)
        sig = self.lsh.signature(x)
        bucket_keys = self.lsh.band_buckets(sig)

        for bk in bucket_keys:
            bucket = self.buckets[bk]

            bucket.add(
                BucketEvent(
                    sensor_id=r.sensor_id,
                    region=r.region,
                    value=value,
                    conf=conf,
                    score=score,
                    t=r.t,
                )
            )
            bucket.expire(r.t)

            # Cluster detection rule: enough events AND enough total severity
            if bucket.count >= self.min_cluster_size and bucket.sum_score >= self.min_cluster_score:
                alerts.append(
                    {
                        "type": "cluster_anomaly",
                        "region": r.region,
                        "bucket": bk,
                        "count": bucket.count,
                        "sum_score": bucket.sum_score,
                        "members": bucket.members(),
                        "note": "LSH bucket collision indicates a potential anomalous cluster",
                    }
                )

        return alerts

    # -------------------------
    # Public API: online processing (record-by-record)
    # -------------------------
    def process(self, r: Reading) -> List[Dict[str, Any]]:
        """Process one reading and return alerts.

        This is the true *online* streaming method (event-by-event).

        Important limitation:
          - Missing values can only use *past* context (already-ingested readings).
          - If you want "same-minute" imputation, use process_minute_window(...).
        """
        conf = clamp(r.conf, 0.0, 1.0)

        # Missing/corrupted: output a probabilistic reconstruction
        if r.value is None:
            region_obs = self.region_buffers[r.region].recent(r.t)
            mu, sigma = impute_distribution_from_region(
                region_obs=region_obs,
                conf_missing=conf,
                now_t=r.t,
            )
            return [
                {
                    "type": "imputation",
                    "sensor": r.sensor_id,
                    "region": r.region,
                    "mu": mu,
                    "sigma": sigma,
                    "note": "Missing/corrupted value reconstructed as a probability distribution",
                }
            ]

        # Observed value
        value = float(r.value)
        return self._process_observed(r, value=value, conf=conf, update_region_buffer=True)

    # -------------------------
    # Public API: windowed processing (micro-batch)
    # -------------------------
    def process_minute_window(self, readings: Iterable[Reading]) -> List[Dict[str, Any]]:
        """Process a 1-minute window (micro-batch) in two passes.

        This method matches the narrative used in the report:
          - All same-minute observed readings are available as "neighbors" for imputing
            missing values (e.g., S204 uses S205/S206/S210 in R3).

        Streaming note:
          - Windowed streaming (event-time or processing-time windows) is standard and
            still satisfies streaming constraints because state is bounded.
        """
        batch = list(readings)
        alerts: List[Dict[str, Any]] = []

        # Pass 1: ingest all observed values into the region buffers first.
        # This makes region context available for imputing missing values in the same window.
        for r in batch:
            if r.value is None:
                continue
            conf = clamp(r.conf, 0.0, 1.0)
            self.region_buffers[r.region].add(value=float(r.value), conf=conf, t=r.t)

        # Pass 2: now process each record (including imputations).
        for r in batch:
            conf = clamp(r.conf, 0.0, 1.0)

            if r.value is None:
                region_obs = self.region_buffers[r.region].recent(r.t)
                mu, sigma = impute_distribution_from_region(
                    region_obs=region_obs,
                    conf_missing=conf,
                    now_t=r.t,
                )
                alerts.append(
                    {
                        "type": "imputation",
                        "sensor": r.sensor_id,
                        "region": r.region,
                        "mu": mu,
                        "sigma": sigma,
                        "note": "Missing/corrupted value reconstructed as a probability distribution",
                    }
                )
                continue

            # Observed: region buffer already updated in pass 1.
            alerts.extend(
                self._process_observed(
                    r,
                    value=float(r.value),
                    conf=conf,
                    update_region_buffer=False,
                )
            )

        return alerts


# -----------------------------
# 8) Demo: run the exact fragment from the assignment prompt
# -----------------------------

def _print_alert(alert: Dict[str, Any]) -> None:
    """Human-friendly printing."""
    atype = alert.get("type")

    if atype == "imputation":
        print(
            f"[IMPUTE] sensor={alert['sensor']} region={alert['region']} "
            f"Normal(mu={alert['mu']:.2f}, sigma={alert['sigma']:.2f})"
        )
        return

    if atype == "spoof_suspect":
        print(
            f"[SPOOF?] sensor={alert['sensor']} region={alert['region']} "
            f"value={alert['value']} conf={alert['conf']:.2f}"
        )
        return

    if atype == "cluster_anomaly":
        print(
            f"[CLUSTER] region={alert['region']} bucket={alert['bucket']} "
            f"count={alert['count']} sum_score={alert['sum_score']:.2f} members={alert['members']}"
        )
        return

    # Fallback
    print("[ALERT]", alert)


if __name__ == "__main__":
    # Pipeline configuration for a small toy demo.
    # For the *real* system (1.2M sensors), you would increase reservoir_k and cluster thresholds.
    pipe = NHICPipeline(
        reservoir_k=50,
        score_min=1.0,
        min_cluster_size=2,      # lowered for tiny sample data
        min_cluster_score=2.0,   # lowered for tiny sample data
        bucket_window_seconds=60.0,
        region_window_seconds=120.0,
        region_buffer_maxlen=200,
    )

    now = time.time()

    # This is the dataset fragment from the assignment prompt.
    # Missing/corrupted values are represented as None.
    stream = [
        Reading("S201", "R2", 57,   0.91, now),
        Reading("S202", "R2", 141,  0.74, now),
        Reading("S203", "R2", 149,  0.18, now),
        Reading("S204", "R3", None, 0.88, now),  # missing
        Reading("S205", "R3", 152,  0.55, now),
        Reading("S206", "R3", 150,  0.93, now),
        Reading("S207", "R4", 31,   0.97, now),
        Reading("S208", "R4", 29,   0.94, now),
        Reading("S209", "R2", None, 0.12, now),  # corrupted
        Reading("S210", "R3", 153,  0.61, now),
    ]

    # IMPORTANT:
    # The report (and the assignment framing) treats the fragment as "one minute" of data.
    # To impute S204 using same-minute neighbors in R3 (S205/S206/S210), we process this
    # minute as a tiny *window* (two-pass micro-batch).
    print("\n=== Minute-window demo (two-pass; matches the report) ===")
    alerts = pipe.process_minute_window(stream)
    for a in alerts:
        _print_alert(a)

    # If we want to see the strict event-by-event online behaviour (no future context
    # for imputations), we can uncomment this:
    #
    # print("\n=== Strict online demo (record-by-record) ===")
    # for r in stream:
    #     for a in pipe.process(r):
    #         _print_alert(a)

    # CMS queries: approximate anomaly indicators
    print("\n=== Count-Min Sketch queries ===")
    print("CMS R3 anomaly_count ~", pipe.cms.query(("R3", "anomaly_count")))
    print("CMS R3 anomaly_mass  ~", pipe.cms.query(("R3", "anomaly_mass")))

    # Reservoir: inspect sampled items
    sample = pipe.reservoir.items()
    print("\n=== Reservoir sample ===")
    print("Reservoir sample size:", len(sample))
    for it in sample:
        # Show only a few fields for readability
        print(f"  - {it.sensor_id} {it.region} value={it.value} conf={it.conf}")
