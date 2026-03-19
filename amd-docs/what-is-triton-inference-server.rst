.. meta::
  :description: What is Triton Inference Server?
  :keywords: Triton Inference Server, documentation, deep learning, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-triton-inference-server:

********************************************************************
What is Triton Inference Server?
********************************************************************

Triton Inference Server is a highly scalable inference serving platform, developed upstream by
`NVIDIA <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html>`__,
that supports multiple backends and is optimized for production LLM workloads. Triton Inference
Server on ROCm integrates AMD ROCm libraries, optimized kernel backends, and expanded support
for LLM inference, providing efficient execution paths for transformer‑based models.

Features and use cases
====================================================================

.. note::

  The ROCm port of Triton Inference Server is under active development, and some features are not yet available. 
  For the most up-to-date feature support, refer to the ``README`` in the 
  `https://github.com/ROCm/triton-inference-server-server <https://github.com/ROCm/triton-inference-server-server>`__ repository.

Triton Inference Server provides the following key features:

- **Multi-Framework Backends:** Serve models from PyTorch, TensorFlow, ONNX Runtime,
  OpenVINO, and the Python backend, with CPU and GPU execution, including ROCm-enabled
  backends (for example, PyTorch-ROCm, ONNX Runtime ROCm EP) for AMD Instinct GPUs.

- **Dynamic and Sequence Batching:** Automatically batch individual requests to
  improve throughput while meeting latency targets, with sequence batching for
  stateful models such as conversational agents.

- **Concurrent Model Execution and Instances:** Run multiple model instances per
  device to saturate hardware, isolate tenants, and tune latency/throughput trade-offs.

- **Model Repository and Versioning:** Manage models stored on local filesystems or
  object storage (for example, S3, GCS, and Azure Blob), with hot reload, versioning, and a
  model control API for lifecycle management.

- **Ensembles and Pipelines:** Compose end-to-end inference graphs for pre/post-processing
  and multi-stage workflows using ensemble models with DAG-style scheduling.

- **Standard Serving Protocols:** Expose HTTP/REST and gRPC endpoints for online
  inference, with support for streaming and shared-memory IPC for lower overhead.

- **Observability and Tooling:** Export Prometheus metrics, traces, and logs; use
  perf analyzer to size instances, tune batching, and validate SLAs.

- **Cloud-Native Deployment:** Ship as Docker images with Helm charts and Kubernetes
  integrations (for example, KServe), supporting autoscaling, canary rollouts, and
  multi-tenant isolation.

- **Extensibility with Python/Custom Backends:** Implement custom backends and
  Python-based logic to integrate specialized runtimes, tokenizers, or business
  rules alongside framework-native backends.

Triton Inference Server is commonly used in the following scenarios:

- **Real-Time LLM and Generative AI Serving:** Host chat, summarization, and
  code-completion services with dynamic batching and concurrent instances to
  balance latency and throughput.

- **Computer Vision and Multimodal Inference:** Serve image, video, and audio
  models with pre/post-processing pipelines as ensembles.

- **Batch and Offline Scoring:** Run large-scale, cost-efficient batch inference
  jobs integrated with data pipelines and object storage.

- **A/B Testing and Canary Deployments:** Safely roll out new model versions and
  compare performance under production traffic.

- **Multi-Tenant Model Hosting:** Consolidate diverse models and frameworks on a
  shared cluster with per-model resource controls.

- **Edge-to-Cloud Deployments:** Use lightweight containers and standardized
  protocols to deploy consistently across environments.

- **RAG and Workflow Orchestration:** Chain retrieval, ranking, and generation
  steps via ensembles or Python backends for end-to-end applications.

Why Triton Inference Server?
====================================================================

Triton Inference Server is well suited for production ML/AI serving for the following reasons:

- **Unified serving stack across frameworks:** Serve ONNX and third-party models
  (and custom runtimes via Python) on AMD Instinct GPUs without bespoke
  per-framework services.

- **High utilization with batching and concurrency:** Dynamic batching, sequence
  batching, and concurrent instances help meet strict SLAs while maximizing GPU use.

- **ROCm performance and flexibility:** Leverage FP16/BF16 execution and HIP-accelerated
  kernels on Instinct MI300 series GPUs, with portable deployment from on-prem to cloud.

- **Enterprise-ready operations:** Cloud-native packaging, rich telemetry, and model
  lifecycle management streamline production deployments at scale.
