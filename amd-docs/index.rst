.. meta::
  :description: Triton Inference Server documentation
  :keywords: Triton Inference Server, ROCm, documentation, deep learning, framework, GPU

.. _triton-inference-server-documentation-index:

********************************************************************
Triton Inference Server on ROCm documentation
********************************************************************

With Triton Inference Server on ROCm, you can deploy multi‑framework, GPU‑accelerated model serving on
AMD Instinct GPUs, enabling low‑latency, elastic inference for production AI services such as real‑time
large language model chat and high‑throughput computer vision analytics.

`Triton Inference Server <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html>`__ is a highly
scalable inference serving platform that supports multiple backends and is optimized for
production LLM workloads. It delivers optimized performance for many query types, including real time, batched,
ensembles and audio/video streaming. Triton Inference Server on ROCm integrates AMD ROCm libraries, optimized kernel
backends, and expanded support for LLM inference, providing efficient execution paths for transformer‑based
models.

.. note::

  The ROCm port of Triton Inference Server is under active development, and some features are not yet available. 
  For the most up-to-date feature support, refer to the ``README`` in the 
  `https://github.com/ROCm/triton-inference-server-server <https://github.com/ROCm/triton-inference-server-server>`__ repository.

Triton Inference Server is part of the `ROCm-LLMExt toolkit
<https://rocm.docs.amd.com/projects/rocm-llmext/en/docs-26.03/>`__.

The Triton Inference Server public repository is located at `https://github.com/ROCm/triton-inference-server-server <https://github.com/ROCm/triton-inference-server-server>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install Triton Inference Server <install/triton-inference-server-install>`

  .. grid-item-card:: Examples

    * :doc:`Run a Triton Inference Server example <examples/triton-inference-server-examples>`

  .. grid-item-card:: Reference

      * `User guide and API reference (upstream) <https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html>`__

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the :doc:`Licensing <about/license>` page.
