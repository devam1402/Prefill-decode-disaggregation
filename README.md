# Prefill-decode-disaggregation
# Distributed LLM Serving on Kubernetes

> A hands-on project to understand and build disaggregated LLM inference: separate prefill and decode GPU pools, transfer KV cache between workers, and orchestrate the system on Kubernetes.

**Project status:** Design / implementation roadmap  
**Primary learning path:** NVIDIA Dynamo → llm-d → custom Kubernetes operator

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Why Disaggregate Prefill and Decode?](#2-why-disaggregate-prefill-and-decode)
- [3. Architecture](#3-architecture)
- [4. Dynamo vs llm-d](#4-dynamo-vs-llm-d)
- [5. Technology Stack](#5-technology-stack)
- [6. Request Lifecycle](#6-request-lifecycle)
- [7. KV Cache Transfer](#7-kv-cache-transfer)
- [8. Implementation Roadmap](#8-implementation-roadmap)
- [9. Repository Structure](#9-repository-structure)
- [10. Benchmarking Plan](#10-benchmarking-plan)
- [11. Hardware Plan](#11-hardware-plan)
- [12. Reliability and Failure Handling](#12-reliability-and-failure-handling)
- [13. Learning Resources](#13-learning-resources)
- [14. Success Criteria](#14-success-criteria)

---

## 1. Project Overview

The goal is to build and benchmark a distributed LLM serving stack that can run prefill and decode on separate GPU workers, including workers on different physical nodes.

The project is organized around three engineering layers:

1. **Inference runtime and data movement:** NVIDIA Dynamo, vLLM, NIXL and supported transports.
2. **Kubernetes inference orchestration:** llm-d, Gateway API, Envoy and Endpoint Picker (EPP).
3. **Platform engineering:** a custom Go/Kubebuilder operator, Kueue, GPU allocation, multi-tenancy and observability.

The first milestone is not to recreate Dynamo or llm-d. It is to deploy, inspect, measure and understand their components, then add platform capabilities around them.

### Project goals

- Understand the prefill and decode phases of autoregressive inference.
- Deploy an aggregated serving baseline.
- Run disaggregated prefill and decode workers.
- Transfer KV cache between workers.
- Progress from same-host GPU transfer to cross-node transfer.
- Measure the trade-offs introduced by disaggregation.
- Compare Dynamo and llm-d routing and orchestration approaches.
- Build Kubernetes-native lifecycle and resource management.

### Non-goals for the first milestone

- Implementing a new transformer inference engine.
- Writing a custom RDMA transport before validating existing integrations.
- Building a production-grade scheduler before establishing a benchmark baseline.
- Assuming that disaggregation is always faster or cheaper.

---

## 2. Why Disaggregate Prefill and Decode?

LLM inference has two main phases.

### Prefill

- Processes the input prompt.
- Computes attention over the prompt tokens.
- Produces the initial key/value (KV) cache.
- Is generally compute-intensive and benefits from high parallelism.

### Decode

- Generates output tokens autoregressively.
- Reuses the KV cache from previous tokens.
- Is often sensitive to memory bandwidth, scheduling and per-token latency.

In an aggregated server, both phases share the same worker resources. In a disaggregated system, prefill and decode can use separate worker pools.

Potential benefits:

- Scale prompt processing and token generation independently.
- Tune GPU resources for different workload profiles.
- Reduce interference between large prompt-processing bursts and ongoing decode requests.
- Apply different scheduling and autoscaling policies to each pool.

Trade-offs:

- KV cache must move from prefill to decode.
- Network bandwidth and transfer latency become important.
- Separate pools can leave GPU capacity stranded.
- Worker coordination, request state and failure recovery become more complex.

The central question is:

> For which prompt lengths, output lengths, concurrency levels and hardware configurations does the benefit of independent scaling exceed KV-transfer and resource-partitioning overhead?

---

## 3. Architecture

### 3.1 Target architecture

```text
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
```

This is a conceptual target. The exact router, worker protocol, connector and transport depend on the selected runtime and release.

### 3.2 Development stages

| Stage | Topology | Purpose |
|---|---|---|
| A | One GPU, aggregated serving | Establish baseline |
| B | Two GPUs on one host | Understand disaggregation and KV movement |
| C | Two physical GPU nodes | Validate cross-node transfer |
| D | Kubernetes worker pools | Deploy and manage distributed workers |
| E | Multi-tenant platform | Add quotas, scheduling and isolation |

---

## 4. Dynamo vs llm-d

These projects overlap, but they have different centers of gravity.

| Area | NVIDIA Dynamo | llm-d |
|---|---|---|
| Primary orientation | Distributed inference system | Kubernetes-native inference orchestration |
| Prefill/decode disaggregation | Supported | Supported |
| KV-cache transfer | NIXL-based integrations and supported connectors | Integrates with supported KV-transfer backends |
| Routing and scheduling | Runtime, KV-aware routing and planning components | Gateway API, Envoy and EPP |
| Kubernetes | Supported deployment model | Core architectural foundation |
| Main study focus | Runtime, KV movement, GPU-aware serving | Routing, endpoint selection and Kubernetes orchestration |

### Recommended learning order

**First: NVIDIA Dynamo**

Study:
- Distributed serving architecture.
- Worker and frontend roles.
- KV-aware routing.
- KV block management.
- NIXL and supported transfer backends.
- Disaggregated serving deployment and benchmarks.

**Second: llm-d**

Study:
- Gateway API and Envoy integration.
- InferencePool.
- Endpoint Picker (EPP).
- Prefill/decode routing.
- KV-aware scheduling and routing integrations.
- Kubernetes deployment and scaling.

Do not run two independent request routers in front of the same workers in the first implementation. Choose one routing control plane for each experiment and compare the alternatives separately.

---

## 5. Technology Stack

| Layer | Candidate technologies | Responsibility |
|---|---|---|
| Model serving | vLLM | Execute model inference |
| Distributed inference | NVIDIA Dynamo | Coordinate distributed serving |
| Kubernetes orchestration | llm-d | Route and orchestrate inference workloads |
| KV transfer | NIXL and supported connectors | Move KV data between workers |
| Transport | TCP, UCX, RoCE, InfiniBand | Carry data between processes and nodes |
| API and ingress | Gateway API, Envoy | Expose and route requests |
| Scheduling | EPP, Kueue | Endpoint selection and workload admission |
| GPU management | NVIDIA GPU Operator, device plugin | Expose and manage GPUs |
| Platform controller | Go, Kubebuilder | Manage platform resources |
| Metrics | Prometheus | Collect metrics |
| Dashboards | Grafana | Visualize performance |
| Tracing | OpenTelemetry | Follow requests across components |
| Load testing | GenAI-Perf, k6 or an equivalent tool | Generate repeatable workloads |

**Important:** Feature availability and integration maturity vary by project release. Pin versions and validate the exact compatibility matrix before deployment.

---

## 6. Request Lifecycle

A typical disaggregated request follows this conceptual sequence:

1. A client submits an inference request to the API endpoint.
2. The gateway authenticates and forwards the request.
3. The router selects a prefill worker.
4. The prefill worker processes the prompt and creates KV cache.
5. The system coordinates KV transfer metadata and destination allocation.
6. The KV connector transfers the required cache blocks to a decode worker.
7. The decode worker resumes generation using the transferred KV cache.
8. Generated tokens stream back through the serving frontend to the client.
9. Request and cache resources are released when the request completes or fails.

The transfer is not simply an HTTP upload of a serialized tensor. Correct operation requires compatible model configuration, tensor layout, cache block mapping, memory registration, transfer coordination and lifecycle handling.

---

## 7. KV Cache Transfer

### 7.1 Approximate KV-cache size

For a conventional decoder-only transformer, an approximate per-request KV-cache size is:

```text
KV bytes =
    2
  × number of layers
  × number of KV heads
  × head dimension
  × number of cached tokens
  × bytes per element
```

The factor of two represents Key and Value.

Example:

- Layers: 32
- KV heads: 8
- Head dimension: 128
- Cached tokens: 8,192
- Data type: BF16 (2 bytes)

```text
2 × 32 × 8 × 128 × 8,192 × 2
= 1,073,741,824 bytes
≈ 1 GiB
```

This is an illustrative estimate. Actual memory use depends on architecture, cache layout, block allocation, tensor parallelism, cache data type and implementation overhead.

### 7.2 Transfer concepts to understand

- KV-cache block allocation and ownership.
- Source and destination worker metadata.
- Memory registration and deregistration.
- GPU memory handles and transport compatibility.
- Transfer completion and error reporting.
- Backpressure and concurrent transfers.
- Cache cleanup after cancellation or timeout.
- Model and parallelism configuration compatibility.

### 7.3 Transport progression

Start with the simplest supported transport and progress only after correctness is established.

1. Same-host transfer using a supported local mechanism.
2. Cross-node TCP transfer as a functional baseline, if supported by the chosen connector.
3. RDMA using compatible NICs, drivers, firmware and container configuration.
4. Measure transfer latency and bandwidth under concurrency.

NCCL, UCX, NIXL, NVLink and RDMA are not interchangeable. Each has a distinct role. Select the transport documented for the chosen runtime and deployment.

---

## 8. Implementation Roadmap

### Phase 0 — Environment and version pinning

- [ ] Select a model and record its revision.
- [ ] Select compatible Dynamo, vLLM and connector versions.
- [ ] Record CUDA, driver and GPU Operator versions.
- [ ] Document GPU and network topology.
- [ ] Create reproducible deployment configuration.

**Exit criteria:** A documented environment that can be recreated.

### Phase 1 — Aggregated baseline

- [ ] Deploy one model using vLLM.
- [ ] Expose an OpenAI-compatible endpoint.
- [ ] Generate short-prompt and long-prompt workloads.
- [ ] Measure TTFT, TPOT and throughput.
- [ ] Capture GPU memory and utilization.
- [ ] Save raw benchmark results.

**Exit criteria:** Repeatable baseline benchmark.

### Phase 2 — Same-host disaggregation

- [ ] Deploy independent prefill and decode workers.
- [ ] Configure a supported disaggregated KV connector.
- [ ] Verify model and cache configuration compatibility.
- [ ] Complete a request from prefill through decode.
- [ ] Verify generated output against the baseline.
- [ ] Measure KV transfer time and end-to-end latency.
- [ ] Test cancellation and worker restart behavior.

**Exit criteria:** Correct end-to-end disaggregated inference on one host.

### Phase 3 — Cross-node disaggregation

- [ ] Provision two GPU nodes.
- [ ] Verify GPU visibility on both nodes.
- [ ] Verify network connectivity and bandwidth.
- [ ] Deploy prefill and decode workers on separate nodes.
- [ ] Configure the documented cross-node transport.
- [ ] Verify KV transfer correctness.
- [ ] Benchmark transfer latency and bandwidth.
- [ ] Test network interruption and worker failure.

**Exit criteria:** A reproducible request completes across two physical nodes.

### Phase 4 — Dynamo deep dive

- [ ] Deploy the selected Dynamo release.
- [ ] Trace the request path through its components.
- [ ] Inspect KV-aware routing behavior.
- [ ] Understand KV block management.
- [ ] Inspect NIXL integration and transfer metrics.
- [ ] Compare supported deployment configurations.
- [ ] Run the same benchmark workload.

**Exit criteria:** Documented Dynamo architecture and reproducible benchmark.

### Phase 5 — llm-d deep dive

- [ ] Deploy the selected llm-d release.
- [ ] Configure Gateway API and Envoy.
- [ ] Deploy InferencePool and EPP.
- [ ] Inspect endpoint selection behavior.
- [ ] Test prefill/decode routing.
- [ ] Evaluate supported KV-aware routing integrations.
- [ ] Compare routing and operational behavior.

**Exit criteria:** Documented llm-d deployment and benchmark.

### Phase 6 — Kubernetes platform operator

Build a custom Go/Kubebuilder controller to manage the platform.

Potential custom resource:

```yaml
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
```

This is an illustrative API design, not a drop-in CRD for Dynamo or llm-d. Adapt resource fields to the selected runtime and validated deployment.

Operator responsibilities:

- [ ] Validate custom resources.
- [ ] Create and reconcile worker deployments.
- [ ] Manage service discovery and endpoints.
- [ ] Integrate with Kueue.
- [ ] Enforce namespace-level quotas and policies.
- [ ] Report readiness and failure conditions.
- [ ] Support independent pool scaling.
- [ ] Handle safe rollout and worker replacement.

**Exit criteria:** A custom resource can declaratively manage the lifecycle of the distributed serving deployment.

---

## 9. Repository Structure

```text
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
```

Keep experimental results, deployment manifests and version information in source control. Do not commit credentials, API keys or private cluster configuration.

---

## 10. Benchmarking Plan

Compare aggregated and disaggregated serving using the same model, model revision, hardware, request distribution and measurement procedure.

### 10.1 Metrics

| Metric | Meaning |
|---|---|
| TTFT | Time to first token |
| TPOT | Time per output token |
| Inter-token latency | Time between generated tokens |
| Input throughput | Prompt tokens processed per second |
| Output throughput | Generated tokens per second |
| Request throughput | Completed requests per second |
| KV transfer latency | Time spent transferring KV cache |
| Transfer bandwidth | Effective data-transfer rate |
| GPU utilization | Compute activity on each worker |
| GPU memory | Memory consumed by each pool |
| Goodput | Requests meeting defined latency SLOs |

### 10.2 Workload matrix

| Experiment | Prompt | Output | Concurrency |
|---|---:|---:|---:|
| A | Short | Short | Low |
| B | Long | Short | Low |
| C | Short | Long | Low |
| D | Long | Long | Low |
| E | Long | Short | High |
| F | Mixed | Mixed | High |

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

### 10.3 Key comparison

The final report should answer:

1. Does disaggregation improve TTFT for long prompts?
2. Does KV transfer increase time to first token?
3. How does decode TPOT change under mixed workloads?
4. How much GPU capacity is lost to pool separation?
5. What network bandwidth is required?
6. At what concurrency does disaggregation become beneficial?
7. Which scheduler produces better SLO-compliant goodput?

Do not claim a performance improvement without reproducible measurements.

---

## 11. Hardware Plan

| Experiment | Hardware |
|---|---|
| Aggregated baseline | 1 GPU |
| Same-host disaggregation | 2 GPUs on one host |
| Cross-node disaggregation | 2 GPU nodes, at least 1 GPU per node |
| RDMA validation | 2 GPU nodes with compatible RDMA networking |
| Pool scaling experiments | 4 or more GPUs recommended |

### Cost-conscious execution

1. Develop manifests, operator code and test clients locally.
2. Validate CPU-only components without renting GPUs.
3. Rent one GPU for the aggregated baseline.
4. Rent two GPUs on one host for initial disaggregation.
5. Rent two networked GPU nodes only when ready for cross-node experiments.
6. Collect results and terminate rented resources after each experiment.

A two-GPU setup is enough to learn the core concepts. Two separate GPU nodes are required to demonstrate genuine cross-node KV transfer.

---

## 12. Reliability and Failure Handling

A distributed serving system must handle failures beyond successful inference.

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

---

## 13. Learning Resources

### NVIDIA Dynamo

- Repository: https://github.com/ai-dynamo/dynamo
- Documentation: https://docs.nvidia.com/dynamo/

Focus: distributed inference architecture, routing, KV management, NIXL and deployment.

### llm-d

- Repository: https://github.com/llm-d/llm-d
- Documentation: https://llm-d.ai/docs/

Focus: Gateway API, Envoy, InferencePool, EPP and Kubernetes-native inference orchestration.

### vLLM

- Repository: https://github.com/vllm-project/vllm
- Documentation: https://docs.vllm.ai/

Focus: serving internals, KV connectors and disaggregated prefilling.

### NVIDIA NIXL

- Repository: https://github.com/ai-dynamo/nixl

Focus: transfer abstractions, memory registration and supported transport backends.

Check the current project documentation and compatibility guidance before choosing versions. These projects evolve quickly.

---

## 14. Success Criteria

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
- [ ] Observability for latency, throughput, GPU utilization and KV transfer.

## Final Engineering Objective

Build a system that can answer, with measurements rather than assumptions:

> When should an inference platform use aggregated serving, and when should it disaggregate prefill and decode across GPU nodes?

The strongest outcome is not merely deploying two inference frameworks. It is understanding the data path, scheduling decisions, GPU and network bottlenecks, operational trade-offs and failure modes well enough to design and operate a distributed inference platform.
"""
