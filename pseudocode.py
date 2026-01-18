INITIALISE:
  reservoir = WeightedReservoirSampler(k)
  cms       = CountMinSketch(width, depth)
  lsh       = RandomHyperplaneLSH(num_bits, band_size)
  buckets   = HashMap bucket_id -> SlidingWindowBucketState
  region_buffer = HashMap region -> sliding window of recent valid readings

FOR each incoming record r in stream:
  (id, region, value, conf, t) = r

  IF value is missing or corrupted:
      imputed_dist = ImputeValue(region_buffer[region], conf, t)
      store (id, region, imputed_dist, t)  // or sample and continue
      CONTINUE

  score = AnomalyScore(value, conf, region)

  // keep region history for future imputation
  region_buffer[region].append((value, conf, t))

  IF score < SCORE_MIN:
      CONTINUE

  // 1) Weighted sampling for detailed inspection
  sample_weight = SamplingWeight(score, conf, region)
  reservoir.update(item=r, weight=sample_weight)

  // 2) Approximate counting of anomalies by sketch
  cms.update(key=(region, "anomaly_count"), increment=1)
  cms.update(key=(region, "anomaly_mass"),  increment=score)

  // 3) LSH for cluster detection
  feature_vec = MakeFeatures(value, conf, region, region_buffer)
  signature   = lsh.signature(feature_vec)
  band_keys   = lsh.band_buckets(signature)

  FOR each b in band_keys:
      buckets[b].add_event(id, region, value, conf, score, t)
      buckets[b].expire_old(t)

      IF buckets[b].count >= MIN_CLUSTER_SIZE AND buckets[b].sum_score >= MIN_CLUSTER_SCORE:
          EMIT alert: "Cluster anomaly", region, sensors=buckets[b].members

  // 4) Spoof suspicion heuristic (example)
  IF value > 145 AND conf < 0.2 AND not in_any_large_cluster(id, t):
      EMIT alert: "Likely spoofed spike", sensor=id, region=region



FUNCTION AnomalyScore(value, conf, region):
  base_threshold = 145

  // risk multiplier (high-risk zones matter more)
  risk = 1.0
  IF region in {R2, R3}: risk = 1.35

  // confidence weight: suppress low-confidence spikes
  // example: squashed mapping (prevents conf=0.1 from contributing much)
  conf_w = conf^2

  // magnitude above threshold
  exceed = max(0, value - base_threshold)

  RETURN risk * conf_w * exceed




FUNCTION reservoir_update(item, weight):
  u = Uniform(0, 1)
  key = u^(1/weight)              // larger key => more likely to be kept

  IF reservoir.size < k:
      reservoir.insert(key, item)
  ELSE IF key > reservoir.min_key:
      reservoir.pop_min()
      reservoir.insert(key, item)


FUNCTION cms_update(key, increment):
  FOR i in 1..d:
     idx = hash_i(key) mod width
     table[i][idx] += increment

FUNCTION cms_query(key):
  est = +infinity
  FOR i in 1..d:
     idx = hash_i(key) mod width
     est = min(est, table[i][idx])
  RETURN est


FUNCTION lsh_signature(x):
  bits = []
  FOR each random vector r_j:
      bits[j] = 1 if dot(r_j, x) >= 0 else 0
  RETURN bits

FUNCTION band_buckets(bits):
  split bits into bands of size B
  FOR each band i:
      band_key = hash(i, band_bits)
      yield band_key


FUNCTION ImputeValue(region_history, conf_missing, t_now):
  // region_history contains tuples (value, conf, t)

  N = select recent valid neighbors within time window W
  IF N is empty:
     RETURN prior distribution (e.g., Normal(mu0, sigma0))

  // time decay + confidence weights
  w_i = conf_i * exp(-(t_now - t_i)/tau)

  mu = sum(w_i * value_i) / sum(w_i)
  var = sum(w_i * (value_i - mu)^2) / sum(w_i)

  // inflate uncertainty if the missing sensor confidence is low
  noise = base_noise + (1 - conf_missing) * extra_noise
  sigma2 = var + noise^2

  RETURN Normal(mu, sigma2)


