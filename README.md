# Prefill-decode-disaggregation

A hands-on project to understand and build disaggregated LLM
inference: separate prefill and decode GPU pools, transfer KV cache
between workers, and orchestrate the system on Kubernetes.

Project status: Design / implementation roadmap
Primary learning path: NVIDIA Dynamo → llm-d → custom Kubernetes
operator
Table of Contents
- 1. Project Overview
- 2. Why Disaggregate Prefill and
  Decode?
- 3. Expected Regimes: When to Disaggregate
- 4. Model and Architecture Choices
- 5. Architecture
- 6. Dynamo vs llm-d
- 7. Technology Stack
- 8. Request Lifecycle
- 9. KV Cache Transfer
- 10. Scheduling Considerations
- 11. Implementation Roadmap
- 12. Repository Structure
- 13. Benchmarking Plan
- 14. Observability and Diagnosis
- 15. Hardware Plan
- 16. Reliability and Failure Handling
- 17. Learning Resources
- 18. Success Criteria
1. Project Overview
The goal is to build and benchmark a distributed LLM serving stack that
can run prefill and decode on separate GPU workers, including workers on
different physical nodes.

The project is organized around three engineering layers:
1. Inference runtime and data movement: NVIDIA Dynamo, vLLM, NIXL
   and supported transports.
2. Kubernetes inference orchestration: llm-d, Gateway API, Envoy
   and Endpoint Picker (EPP).
3. Platform engineering: a custom Go/Kubebuilder operator, Kueue,
   GPU allocation, multi-tenancy and observability.
The first milestone is not to recreate Dynamo or llm-d. It is to deploy,
inspect, measure and understand their components, then add platform
capabilities around them.
Project goals
- Understand the prefill and decode phases of autoregressive
  inference.
- Deploy an aggregated serving baseline.
- Run disaggregated prefill and decode workers.
- Transfer KV cache between workers.
- Progress from same-host GPU transfer to cross-node transfer.
- Measure the trade-offs introduced by disaggregation.
- Compare Dynamo and llm-d routing and orchestration approaches.
- Build Kubernetes-native lifecycle and resource management.
Non-goals for the first milestone
- Implementing a new transformer inference engine.
- Writing a custom RDMA transport before validating existing
  integrations.
- Building a production-grade scheduler before establishing a
  benchmark baseline.
- Assuming that disaggregation is always faster or cheaper.
2. Why Disaggregate Prefill and Decode?
LLM inference has two main phases.
Prefill
- Processes the input prompt.
- Computes attention over the prompt tokens.
- Produces the initial key/value (KV) cache.
- Is generally compute-intensive and benefits from high parallelism.
Decode
- Generates output tokens autoregressively.
- Reuses the KV cache from previous tokens.
- Is often sensitive to memory bandwidth, scheduling and per-token
  latency.
In an aggregated server, both phases share the same worker resources. In
a disaggregated system, prefill and decode can use separate worker
pools.

Potential benefits:
- Scale prompt processing and token generation independently.
- Tune GPU resources for different workload profiles.
- Reduce interference between large prompt-processing bursts and
  ongoing decode requests.
- Apply different scheduling and autoscaling policies to each pool.
Trade-offs:
- KV cache must move from prefill to decode.
- Network bandwidth and transfer latency become important.
- Separate pools can leave GPU capacity stranded.
- Worker coordination, request state and failure recovery become more
  complex.
The central question is:
For which prompt lengths, output lengths, concurrency levels and
hardware configurations does the benefit of independent scaling exceed
KV-transfer and resource-partitioning overhead?

3. Expected Regimes: When to Disaggregate
Disaggregation is a workload-dependent optimization, not a default deployment mode. Use the following table as an initial decision guide, then validate it with the target model, GPU topology, network and SLO.
Workload or constraint	Initial preference	Reason
p99 inter-token latency (ITL) misses its SLO under mixed prompt/decode load	Evaluate disaggregation first	Separating prompt bursts from active decode streams can reduce decode interference.
Long prompts, high concurrency and a fast interconnect	Strongly consider disaggregation	Transfer overhead may be small relative to prefill work and the benefit of independent pool sizing.
TTFT is the hard constraint, outputs are short, or concurrency is moderate on commodity networking	Stay aggregated until measured	KV transfer is on the critical path before decode can start.
High prefix-cache hit rate (shared system prompts, multi-turn sessions or RAG prefixes)	Aggregated or carefully designed hybrid	Less prefill work remains to amortize transfer overhead. Cache locality and only transferring missing KV can change the result.
Heterogeneous GPUs are available	Strongly consider disaggregation	Compute-optimized prefill and memory-bandwidth-optimized decode can be a compelling resource match.


These are hypotheses for experiment design, not guarantees. The exact break-even point depends on model architecture, cache layout, prompt/output distribution, batching, concurrency, GPU topology and transport implementation. Published benchmarks are not directly comparable unless those conditions match.
KV-transfer time-of-flight: do the math first
The example in KV Cache Transfer produces approximately 1 GiB of KV data for one request. A useful lower-bound estimate is:
ideal transfer time (seconds) = KV bytes / effective link bytes per second
For a 1 GiB payload, ignoring protocol overhead, contention, registration and synchronization:
Effective one-way bandwidth	Ideal time for 1 GiB
10 GB/s	~107 ms
25 GB/s	~43 ms
50 GB/s	~21 ms
100 GB/s	~11 ms


Treat these as arithmetic lower bounds, not measured platform performance. A nominal 100/400 Gb/s link does not guarantee that rate for an individual transfer. Real results depend on effective bandwidth, topology, concurrency, transport and connector overhead. Ordinary Ethernet can add hundreds of milliseconds or more when effective throughput is low or links are contended. PCIe is an intra-node interconnect, not a substitute name for the network path.

A simple decision check:
1. Estimate KV bytes for the actual model, prompt length and cache dtype.
2. Measure effective transfer bandwidth on the target topology.
3. Estimate transfer time and compare it with the TTFT and ITL budgets.
4. Benchmark end-to-end; do not decide from link line rate alone.
4. Model and Architecture Choices
Starter model
Use Qwen2.5-7B-Instruct as the initial benchmark model, with a pinned model revision. It is a practical 7B-class starting point with grouped-query attention (GQA), broad tooling support and enough prompt-processing work to make the prefill/decode trade-off more visible than it would be with a tiny model.

A second model for comparison can be Llama 3.1 8B Instruct. Qwen3-8B is also a reasonable alternative if the selected runtime and model revision are validated together.

Avoid using a 0.6B-class model as the only performance comparison: its prefill phase can be so fast that it hides the interference and transfer trade-offs the project is meant to study.

Whether a model fits on one or two GPUs depends on weight dtype/quantization, KV-cache capacity, context length, batch size, parallelism and runtime overhead. Record those details rather than describing a model as universally “comfortable” on a given GPU.
Architecture details that change the economics
- GQA vs MHA: GQA uses fewer KV heads than query heads, reducing KV-cache bytes compared with full multi-head attention at the same layer count, head dimension, token count and cache dtype. The formula in the next section uses number of KV heads, not query heads.
- KV-cache quantization: FP8 or another supported 8-bit KV format can approximately halve cache bytes compared with BF16/FP16. Validate quality, kernel support, cache compatibility and transfer-path support for the selected stack.
- Prefix caching: When the destination can reuse matching prefix blocks, only the missing suffix may need to be computed or transferred, depending on the runtime's cache-sharing design. High prefix reuse can make disaggregation less attractive if the transfer cost remains while prefill work falls.
- MLA architectures: Multi-head Latent Attention (MLA), used by DeepSeek-family models, stores a compressed latent representation rather than the conventional full K/V representation. This can reduce KV memory and transfer volume substantially—sometimes by an order of magnitude or more versus a comparable conventional cache—but the exact ratio is architecture- and implementation-dependent.
- Heterogeneous pools: Treat compute-optimized prefill + memory-bandwidth-optimized decode as a first-class reason to disaggregate. It may be valuable even when a homogeneous cluster would not benefit.
Pin the model revision, tokenizer revision, attention/cache implementation and cache dtype in every benchmark record.
5. Architecture
5.1 Target architecture
                         Clients
                            |
                            v
                    OpenAI-compatible API
                            |
                            v
                     Gateway / Envoy
                            |
                            v
                    Inference Router
                            |
                  Scheduling / Admission
                            |
                 +----------+----------+
                 |                     |
                 v                     v
           PREFILL POOL            DECODE POOL
          +-------------+         +-------------+
          | Prefill     |         | Decode      |
          | Worker 1    |         | Worker 1    |
          +------+------+         +------+------+
                 |                       ^
                 |                       |
                 v                       |
          KV cache produced              |
                 |                       |
                 v                       |
          KV transfer layer -------------+
             NIXL / supported
             transport backend
                 |
          Network fabric
          TCP / RoCE / InfiniBand
          depending on hardware support

        Kubernetes control and platform layer
        --------------------------------------
        Go Operator | Kueue | GPU Operator
        Worker lifecycle | quotas | placement

        Observability
        --------------
        Prometheus | Grafana | OpenTelemetry
        TTFT / ITL | phase queues | KV transfer | GPU SM / HBM
This is a conceptual target. The exact router, worker protocol,
connector and transport depend on the selected runtime and release.
5.2 Development stages
  Stage   Topology                      Purpose
  A       One GPU, aggregated serving   Establish baseline
B       Two GPUs on one host          Understand disaggregation and KV movement
C       Two physical GPU nodes        Validate cross-node transfer
D       Kubernetes worker pools       Deploy and manage distributed workers
E       Multi-tenant platform         Add quotas, scheduling and isolation
6. Dynamo vs llm-d
These projects overlap, but they have different centers of gravity.
  Area                    NVIDIA Dynamo           llm-d
  Primary orientation     Distributed inference   Kubernetes-native
system                  inference orchestration

  Prefill/decode          Supported               Supported
disaggregation                                  

  KV-cache transfer       NIXL-based integrations Integrates with
and supported           supported KV-transfer
connectors              backends

  Routing and scheduling  Runtime, KV-aware       Gateway API, Envoy and
routing and planning    EPP
components              

  Kubernetes              Supported deployment    Core architectural
model                   foundation
Main study focus        Runtime, KV movement,   Routing, endpoint
GPU-aware serving       selection and
Kubernetes
orchestration
Recommended learning order
First: NVIDIA Dynamo

Study: - Distributed serving architecture. - Worker and frontend
roles. - KV-aware routing. - KV block management. - NIXL and supported
transfer backends. - Disaggregated serving deployment and benchmarks.

Second: llm-d

Study: - Gateway API and Envoy integration. - InferencePool. - Endpoint
Picker (EPP). - Prefill/decode routing. - KV-aware scheduling and
routing integrations. - Kubernetes deployment and scaling.

Do not run two independent request routers in front of the same workers
in the first implementation. Choose one routing control plane for each
experiment and compare the alternatives separately.
7. Technology Stack
  Layer                   Candidate technologies  Responsibility
  Model serving           vLLM                    Execute model inference

  Distributed inference   NVIDIA Dynamo           Coordinate distributed
serving

  Kubernetes              llm-d                   Route and orchestrate
orchestration                                   inference workloads

  KV transfer             NIXL and supported      Move KV data between
connectors              workers

  Transport               TCP, UCX, RoCE,         Carry data between
InfiniBand              processes and nodes

  API and ingress         Gateway API, Envoy      Expose and route
requests

  Scheduling              EPP, Kueue              Endpoint selection and
workload admission

  GPU management          NVIDIA GPU Operator,    Expose and manage GPUs
device plugin           

  Platform controller     Go, Kubebuilder         Manage platform
resources

  Metrics                 Prometheus              Collect metrics

  Dashboards              Grafana                 Visualize performance

  Tracing                 OpenTelemetry           Follow requests across
components
Load testing            GenAI-Perf, k6 or an    Generate repeatable
equivalent tool         workloads
Important: Feature availability and integration maturity vary by
project release. Pin versions and validate the exact compatibility
matrix before deployment.
8. Request Lifecycle
A typical disaggregated request follows this conceptual sequence:
1. A client submits an inference request to the API endpoint.
2. The gateway authenticates and forwards the request.
3. The router selects a prefill worker.
4. The prefill worker processes the prompt and creates KV cache.
5. The system coordinates KV transfer metadata and destination
   allocation.
6. The KV connector transfers the required cache blocks to a decode
   worker.
7. The decode worker resumes generation using the transferred KV cache.
8. Generated tokens stream back through the serving frontend to the
   client.
9. Request and cache resources are released when the request completes
   or fails.
The transfer is not simply an HTTP upload of a serialized tensor.
Correct operation requires compatible model configuration, tensor
layout, cache block mapping, memory registration, transfer coordination
and lifecycle handling.
9. KV Cache Transfer
9.1 Approximate KV-cache size
For a conventional decoder-only transformer, an approximate per-request
KV-cache size is:
KV bytes =
    2
  × number of layers
  × number of KV heads
  × head dimension
  × number of cached tokens
  × bytes per element
The factor of two represents Key and Value.

Example:
- Layers: 32
- KV heads: 8
- Head dimension: 128
- Cached tokens: 8,192
- Data type: BF16 (2 bytes)
2 × 32 × 8 × 128 × 8,192 × 2
= 1,073,741,824 bytes
≈ 1 GiB
This is an illustrative estimate. Actual memory use depends on
architecture, cache layout, block allocation, tensor parallelism, cache
data type and implementation overhead.
9.2 Transfer concepts to understand
- KV-cache block allocation and ownership.
- Source and destination worker metadata.
- Memory registration and deregistration.
- GPU memory handles and transport compatibility.
- Transfer completion and error reporting.
- Backpressure and concurrent transfers.
- Cache cleanup after cancellation or timeout.
- Model and parallelism configuration compatibility.
9.3 Transport progression
Start with the simplest supported transport and progress only after
correctness is established.
1. Same-host transfer using a supported local mechanism.
2. Cross-node TCP transfer as a functional baseline, if supported by
   the chosen connector.
3. RDMA using compatible NICs, drivers, firmware and container
   configuration.
4. Measure transfer latency and bandwidth under concurrency.
NCCL, UCX, NIXL, NVLink and RDMA are not interchangeable. Each has a
distinct role. Select the transport documented for the chosen runtime
and deployment.
10. Scheduling Considerations
Prefill and decode have different workload shapes: prefill is often bursty, while decode requests occupy workers for a longer period and generate tokens over time.
- Do not assume a 1:1 pool ratio. Sweep prefill-to-decode capacity ratios. A 1:2 or 1:4 prefill:decode starting sweep can be useful for exploration, but it is not a universal production ratio; the right ratio depends on prompt/output distributions, model, GPU and SLO.
- Track queues independently. Record prefill queue depth and wait time separately from decode queue depth and wait time. Aggregate queue metrics can hide one saturated pool.
- Use cache-aware routing when it matters. Prefix/KV-aware routing can avoid recomputation or unnecessary movement. Session stickiness can improve cache reuse, but can also create hot workers and uneven load.
- Balance locality against load. A cache-local worker is not always the best choice if it is overloaded. Routing should account for both cache locality and current queue/capacity.
- Treat cancellation as resource management. Cancellation, timeout and worker failure must release partial KV allocations and transfer state. See Reliability and Failure Handling.
The scheduler should expose enough signals to explain why a request was sent to a particular prefill and decode worker.
11. Implementation Roadmap
Phase 0 --- Environment and version pinning
- [ ] Select a model and record its revision.
- [ ] Select compatible Dynamo, vLLM and connector versions.
- [ ] Record CUDA, driver and GPU Operator versions.
- [ ] Document GPU and network topology.
- [ ] Create reproducible deployment configuration.
Exit criteria: A documented environment that can be recreated.
Phase 1 --- Aggregated baseline
- [ ] Deploy one model using vLLM.
- [ ] Expose an OpenAI-compatible endpoint.
- [ ] Generate short-prompt and long-prompt workloads.
- [ ] Measure TTFT, TPOT and throughput.
- [ ] Capture GPU memory and utilization.
- [ ] Save raw benchmark results.
Exit criteria: Repeatable baseline benchmark.
Phase 2 --- Same-host disaggregation
- [ ] Deploy independent prefill and decode workers.
- [ ] Configure a supported disaggregated KV connector.
- [ ] Verify model and cache configuration compatibility.
- [ ] Complete a request from prefill through decode.
- [ ] Verify generated output against the baseline.
- [ ] Measure KV transfer time and end-to-end latency.
- [ ] Test cancellation and worker restart behavior.
Exit criteria: Correct end-to-end disaggregated inference on one
host.
Phase 3 --- Cross-node disaggregation
- [ ] Provision two GPU nodes.
- [ ] Verify GPU visibility on both nodes.
- [ ] Verify network connectivity and bandwidth.
- [ ] Deploy prefill and decode workers on separate nodes.
- [ ] Configure the documented cross-node transport.
- [ ] Verify KV transfer correctness.
- [ ] Benchmark transfer latency and bandwidth.
- [ ] Test network interruption and worker failure.
Exit criteria: A reproducible request completes across two physical
nodes.
Phase 4 --- Dynamo deep dive
- [ ] Deploy the selected Dynamo release.
- [ ] Trace the request path through its components.
- [ ] Inspect KV-aware routing behavior.
- [ ] Understand KV block management.
- [ ] Inspect NIXL integration and transfer metrics.
- [ ] Compare supported deployment configurations.
- [ ] Run the same benchmark workload.
Exit criteria: Documented Dynamo architecture and reproducible
benchmark.
Phase 5 --- llm-d deep dive
- [ ] Deploy the selected llm-d release.
- [ ] Configure Gateway API and Envoy.
- [ ] Deploy InferencePool and EPP.
- [ ] Inspect endpoint selection behavior.
- [ ] Test prefill/decode routing.
- [ ] Evaluate supported KV-aware routing integrations.
- [ ] Compare routing and operational behavior.
Exit criteria: Documented llm-d deployment and benchmark.
Phase 6 --- Kubernetes platform operator
Build a custom Go/Kubebuilder controller to manage the platform.

Potential custom resource:
apiVersion: inference.example.com/v1alpha1
kind: DistributedInferenceService
metadata:
  name: qwen-serving
spec:
  model:
    name: Qwen
    # Pin the actual model revision in the deployment.

  runtime:
    name: dynamo

  prefill:
    replicas: 1
    resources:
      nvidia.com/gpu: 1

  decode:
    replicas: 1
    resources:
      nvidia.com/gpu: 1

  kvTransfer:
    enabled: true
    transport: tcp

  scheduling:
    queue: inference-queue
This is an illustrative API design, not a drop-in CRD for Dynamo or
llm-d. Adapt resource fields to the selected runtime and validated
deployment.

Operator responsibilities:
- [ ] Validate custom resources.
- [ ] Create and reconcile worker deployments.
- [ ] Manage service discovery and endpoints.
- [ ] Integrate with Kueue.
- [ ] Enforce namespace-level quotas and policies.
- [ ] Report readiness and failure conditions.
- [ ] Support independent pool scaling.
- [ ] Handle safe rollout and worker replacement.
Exit criteria: A custom resource can declaratively manage the
lifecycle of the distributed serving deployment.
12. Repository Structure
distributed-llm-serving/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── design-decisions.md
│   ├── dynamo-notes.md
│   ├── llm-d-notes.md
│   ├── kv-cache-transfer.md
│   ├── networking.md
│   └── benchmark-analysis.md
│
├── deployments/
│   ├── aggregated/
│   ├── dynamo/
│   ├── llm-d/
│   └── kubernetes/
│
├── operator/
│   ├── api/
│   ├── internal/
│   ├── config/
│   └── controllers/
│
├── workloads/
│   ├── prefill/
│   ├── decode/
│   └── clients/
│
├── networking/
│   ├── tcp/
│   ├── rdma/
│   └── diagnostics/
│
├── benchmarks/
│   ├── workloads/
│   ├── scripts/
│   ├── results/
│   └── reports/
│
├── observability/
│   ├── prometheus/
│   ├── grafana/
│   └── dashboards/
│
├── tests/
│   ├── integration/
│   ├── failure/
│   └── performance/
│
└── scripts/
    ├── setup.sh
    ├── validate-environment.sh
    └── collect-results.sh
Keep experimental results, deployment manifests and version information
in source control. Do not commit credentials, API keys or private
cluster configuration.
13. Benchmarking Plan
Compare aggregated and disaggregated serving using the same model, model
revision, hardware, request distribution and measurement procedure.
13.1 Metrics
  Metric                Meaning
  TTFT                  Time to first token
TPOT                  Time per output token
Inter-token latency   Time between generated tokens
Input throughput      Prompt tokens processed per second
Output throughput     Generated tokens per second
Request throughput    Completed requests per second
KV transfer latency   Time spent transferring KV cache
Transfer bandwidth    Effective data-transfer rate
GPU utilization       Compute activity on each worker
GPU memory            Memory consumed by each pool
Goodput               Requests meeting defined latency SLOs
13.2 Workload matrix
  Experiment     Prompt   Output   Concurrency
  A               Short    Short           Low
B                Long    Short           Low
C               Short     Long           Low
D                Long     Long           Low
E                Long    Short          High
F               Mixed    Mixed          High

Record the following for every run:
- Model and model revision.
- Runtime and connector versions.
- GPU type and count.
- GPU topology.
- Node and network topology.
- Prompt and output token distributions.
- Concurrency and request rate.
- Warm-up and measurement duration.
- Percentile latency values.
- Raw logs and benchmark output.
13.3 Key comparison
Record even the first baseline run. A single honest data point is more useful than an empty results directory.
Run ID	Mode	Model / revision	GPUs	Input / output	Concurrency	TTFT p50/p95/p99	ITL p50/p95/p99	Output tok/s	Notes
baseline-001	Aggregated	TBD	1	2k / 256	8	TBD	TBD	TBD	First baseline placeholder


Replace TBD only with measured values and link the raw result file. Do not treat this template row as a benchmark result.

The final report should answer:
1. Does disaggregation improve TTFT for long prompts?
2. Does KV transfer increase time to first token?
3. How does decode TPOT change under mixed workloads?
4. How much GPU capacity is lost to pool separation?
5. What network bandwidth is required?
6. At what concurrency does disaggregation become beneficial?
7. Which scheduler produces better SLO-compliant goodput?
Do not claim a performance improvement without reproducible
measurements.
14. Observability and Diagnosis
The goal is not simply to collect generic GPU metrics. Instrument the request path so that a latency regression can be attributed to queueing, compute, transfer, decode or routing.
Minimum useful metrics
Area	Metrics to collect	Why it matters
Prefill	Queue time, queue depth, prefill compute time, input tokens/sec	Separates admission delay from prompt-processing cost.
KV transfer	Transfer size, latency p50/p95/p99, achieved bandwidth, failures/timeouts	Shows whether data movement dominates TTFT.
Decode	Queue time, TPOT, ITL p50/p95/p99, output tokens/sec	Reveals decode saturation and user-visible streaming latency.
GPU compute	Per-pool SM utilization and HBM bandwidth utilization	Distinguishes compute-bound from memory-bound workers.
KV capacity	KV-cache utilization and free blocks per pool	Helps explain cache pressure, preemption and admission failures.
Transfer pressure	Concurrent transfers, queued transfers and backpressure events	Reveals contention in the KV movement path.
Routing	Selected prefill/decode worker, queue depth at selection, cache hit/locality signal	Makes scheduling decisions explainable.


Diagnosis map
Symptom	Likely cause to investigate	First action
High TTFT + high prefill utilization	Prefill saturation, batching or insufficient prefill capacity	Inspect prefill queue/compute split; tune batching or scale prefill.
High TTFT + low prefill utilization + high transfer latency	Network, connector, memory registration or transfer contention	Inspect transfer size, achieved bandwidth, concurrent transfers and transport errors.
High ITL/TPOT + high decode KV-cache utilization	Decode capacity or cache pressure	Inspect free KV blocks, active sequences and decode queue; scale decode or tune admission/concurrency.
Both pools under-utilized but latency is high	Router, admission control, queueing or coordination overhead	Trace request timestamps across gateway, router, prefill, transfer and decode.
Good average latency but poor p99	Bursts, noisy neighbors, uneven routing or transfer tail latency	Compare p50/p95/p99 by phase and inspect per-worker skew.


Export and scrape strategy
Some serving engines already expose useful request, scheduler, cache and latency metrics. vLLM exposes Prometheus metrics, while transfer connectors and NIXL integrations may expose additional transfer statistics depending on version and backend. Do not assume every metric has a stable name across releases.
- Scrape runtime metrics from every prefill and decode worker.
- Add transfer metrics at the connector boundary if the runtime does not export them.
- Attach request IDs or trace IDs to logs/traces, but avoid high-cardinality Prometheus labels.
- Keep labels bounded: pool, worker, model, status and transport are usually more useful than per-request labels.
- Record histogram buckets that can resolve the SLO thresholds you care about.
- Dashboard prefill and decode pools separately, then add an end-to-end request view.
15. Hardware Plan
  Experiment                  Hardware
  Aggregated baseline         1 GPU
Same-host disaggregation    2 GPUs on one host
Cross-node disaggregation   2 GPU nodes, at least 1 GPU per node
RDMA validation             2 GPU nodes with compatible RDMA networking
Pool scaling experiments    4 or more GPUs recommended
Cost-conscious execution
1. Develop manifests, operator code and test clients locally.
2. Validate CPU-only components without renting GPUs.
3. Rent one GPU for the aggregated baseline.
4. Rent two GPUs on one host for initial disaggregation.
5. Rent two networked GPU nodes only when ready for cross-node
   experiments.
6. Collect results and terminate rented resources after each
   experiment.
A two-GPU setup is enough to learn the core concepts. Two separate GPU
nodes are required to demonstrate genuine cross-node KV transfer.
16. Reliability and Failure Handling
A distributed serving system must handle failures beyond successful
inference.

Test these scenarios:
- Prefill worker crashes before KV transfer.
- Decode worker crashes during generation.
- KV transfer times out.
- Network connection is interrupted.
- Client cancels a request.
- A worker becomes unhealthy.
- A GPU runs out of memory.
- A rollout replaces workers with active requests.
- A node becomes unavailable.
For every failure, document:
- Expected behavior.
- Observed behavior.
- Request outcome.
- Resource cleanup.
- Recovery time.
- Metrics and logs available for diagnosis.
17. Learning Resources
NVIDIA Dynamo
- Repository: https://github.com/ai-dynamo/dynamo
- Documentation: https://docs.nvidia.com/dynamo/
Focus: distributed inference architecture, routing, KV management, NIXL
and deployment.
llm-d
- Repository: https://github.com/llm-d/llm-d
- Documentation: https://llm-d.ai/docs/
Focus: Gateway API, Envoy, InferencePool, EPP and Kubernetes-native
inference orchestration.
vLLM
- Repository: https://github.com/vllm-project/vllm
- Documentation: https://docs.vllm.ai/
Focus: serving internals, KV connectors and disaggregated prefilling.
NVIDIA NIXL
- Repository: https://github.com/ai-dynamo/nixl
Focus: transfer abstractions, memory registration and supported
transport backends.

Check the current project documentation and compatibility guidance
before choosing versions. These projects evolve quickly.
18. Success Criteria
The project is complete when it demonstrates:
- [ ] A reproducible aggregated inference baseline.
- [ ] Correct same-host disaggregated inference.
- [ ] Correct cross-node KV-cache transfer.
- [ ] Documented transport configuration and limitations.
- [ ] Independent prefill and decode worker pools.
- [ ] A reproducible Dynamo deployment and evaluation.
- [ ] A reproducible llm-d deployment and evaluation.
- [ ] Benchmark comparisons with raw results.
- [ ] Failure-handling tests.
- [ ] Kubernetes lifecycle management through a custom operator.
- [ ] Observability for latency, throughput, GPU utilization and KV
  transfer.
Final Engineering Objective
Build a system that can answer, with measurements rather than
assumptions:
When should an inference platform use aggregated serving, and when
should it disaggregate prefill and decode across GPU nodes?

The strongest outcome is not merely deploying two inference frameworks.
It is understanding the data path, scheduling decisions, GPU and network
bottlenecks, operational trade-offs and failure modes well enough to
design and operate a distributed inference platform.
Repository Status and Next Action
Current status: Design and implementation plan. No performance claims are made until benchmark results are recorded.

The repository skeleton includes placeholders for deployment manifests, operator code, benchmark data, observability assets and design notes. Start with [Phase 0](docs/phase-0-checklist.md), fill in the exact environment versions, then run the aggregated baseline before enabling disaggregation.
