.. meta::
  :description: Triton Inference Server examples
  :keywords: Triton Inference Server, programming, ROCm, example, sample, tutorial, MI300X, MI355X, example, model repository, ONNX Runtime, perf_analyzer

.. _run-a-triton-inference-server-example:

********************************************************************
Run a Triton Inference Server example
********************************************************************

This example shows how to deploy and run a simple ONNX model using Triton Inference Server with ROCm on AMD Instinct MI300X or MI355X GPUs.

1. Prepare a Triton model repository with a simple ONNX model.

   Create a minimal ONNX model and a Triton model repository layout that uses the ONNX Runtime backend with the ROCm execution provider.

   .. code-block:: bash

      # Create a working directory and model repository
      mkdir -p $PWD/triton_model_repo/finalnet/1
      cd $PWD/triton_model_repo

      # Write a Triton model configuration for ONNX Runtime + ROCm
      cat > finalnet/config.pbtxt << 'EOF'
      name: "finalnet"
      platform: "onnxruntime_onnx"
      max_batch_size: 8

      input [
        {
          name: "input"
          data_type: TYPE_FP32
          dims: [ 1024 ]
        }
      ]

      output [
        {
          name: "output"
          data_type: TYPE_FP32
          dims: [ 1024 ]
        }
      ]

      instance_group [
        {
          kind: KIND_GPU
          count: 1
          gpus: [ 0 ]
        }
      ]

      optimization {
        execution_accelerators {
          gpu_execution_accelerator: [
            { name: "rocm" }
          ]
        }
      }
      EOF

   Save the following Python script to generate a simple identity ONNX model that matches the configuration above.

   .. code-block:: python

      # save as make_identity_model.py (run with: python make_identity_model.py)
      import onnx
      from onnx import helper, TensorProto

      input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1024])
      output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1024])

      node = helper.make_node("Identity", inputs=["input"], outputs=["output"])
      graph = helper.make_graph([node], "IdentityGraph", [input_tensor], [output_tensor])
      model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])

      onnx.checker.check_model(model)
      onnx.save(model, "finalnet/1/model.onnx")

   .. code-block:: bash

      # Generate the ONNX model file in the model repository
      python make_identity_model.py
      ls -R .

2. Start Triton Inference Server (ROCm-enabled) with your model repository.

   Use a ROCm-enabled Triton container and pass through the AMD GPU devices. The example below assumes a ROCm 7.2 + Triton 25.12 image tag.

   .. code-block:: bash

      # Pull the ROCm-enabled Triton Server image
      docker pull rocm/tritoninferenceserver:tritoninferenceserver-25.12.amd1_rocm7.2_ubuntu24.04_py3.12

      # Launch Triton Server, mapping your model repository
      docker run --rm -it --net=host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --shm-size=8g --ulimit memlock=-1 --ulimit stack=67108864 \
        -e HIP_VISIBLE_DEVICES=0 \
        -v $PWD:/models \
        rocm/tritoninferenceserver:tritoninferenceserver-25.12.amd1_rocm7.2_ubuntu24.04_py3.12 \
        tritonserver --model-repository=/models --log-verbose=1

   Keep this terminal running. You should see Triton load the "finalnet" model and report that the server is ready on HTTP port 8000 and gRPC port 8001.

3. Send an inference request with ``perf_analyzer``.

   In a separate terminal, you can use Triton’s ``perf_analyzer`` to validate and measure performance. If perf_analyzer is available in your image, run:

   .. code-block:: bash

      docker run --rm -it --net=host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --security-opt seccomp=unconfined \
        -v $(pwd):/workspace \
        rocm/tritonserver:25.12-rocm7.2 \
        perf_analyzer -m finalnet -b 8 --concurrency-range 1:8 --input-data zero --shape input:1024

   If your server image does not include perf_analyzer, use a Triton SDK/clients image that provides it, or install the Triton Python client and run a small test script as shown below.

4. (Alternative) Send a request via Triton’s Python client.

   Save the following Python client and run it in an environment with ``tritonclient[http]`` installed (using ``pip install tritonclient[http]``).

   .. code-block:: python

      # save as triton_client_test.py
      import numpy as np
      import tritonclient.http as httpclient

      model_name = "finalnet"
      url = "localhost:8000"

      client = httpclient.InferenceServerClient(url=url, verbose=False)

      # Prepare input data (batch size 8, vector of length 1024)
      batch_size = 8
      inp = np.zeros((batch_size, 1024), dtype=np.float32)

      input0 = httpclient.InferInput("input", inp.shape, "FP32")
      input0.set_data_from_numpy(inp)

      output0 = httpclient.InferRequestedOutput("output")

      result = client.infer(model_name=model_name, inputs=[input0], outputs=[output0])
      out = result.as_numpy("output")
      print("Output shape:", out.shape)
      print("First row (should be zeros):", out[0][:8])

   Then run the script:

   .. code-block:: bash

      python triton_client_test.py


