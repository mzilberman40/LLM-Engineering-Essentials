# Topic 4. Self-deployed LLMs

**The course is under construction, with new materials appearing regularly.**

## Contents

* **Transformer internals**

  This part is for those who are curious about what's inside a transformer.

  Start by the videos by [Tatiana Gaintseva](https://www.linkedin.com/in/tgaintseva/) to understand the overall transformer structure and the QKV attention:

  **[1. Seq2seq](https://youtu.be/z-pebP9NuC4)**
  
  **[2. Attention Mechanism](https://youtu.be/e-xQKzi2Hxc)**
  
  **[3a. Transformer Encoder Architecture](https://youtu.be/u_hxAuShuZQ)**
  
  **[3b. Multi-head attention and positional encoding](https://youtu.be/eoQTzGi1BkQ)**
  
  **[3c. Transformer decoder](https://youtu.be/W4nJnW9R3IE)**
  
  **[3d. QKV attention](https://youtu.be/1oMAF55sRls)**

  Continue with the [**long read telling about the state-of-the-art of transformer components**](https://nebius-academy.github.io/knowledge-base/transformer-architectures/).

* **4.1. Open-source LLMs and the Hugging Face Ecosystem** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic4/4.1_open_source_models.ipynb)

  Learn how to use open-source models from Hugging Face - from simple inference to multimodality and tool usage.

* **4.2. Dissecting an LLM** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic4/4.2_dissecting_an_llm.ipynb)
  
  Take a look into internals of several LLMs and learn to use Pytorch hooks to assess and modify hidden states of an LLM on the fly.

* [**LLM Inference Essentials**](https://nebius-academy.github.io/knowledge-base/llm-inference-essentials/)

  Check this long read to start looking at LLM inference from an AI engineer's point of view, understanding inference metrics and efficiency issues. And here is a demonstration notebook:

  **4.3. LLM Inference Metrics** [colab link](https://colab.research.google.com/github/Nebius-Academy/LLM-Engineering-Essentials/blob/main/topic4/4.3_llm_inference_metrics.ipynb)

## Project part: using an open-source LLM in your service

The project materials are in the [LLMOps Essentials repo](https://github.com/Nebius-Academy/LLMOps-Essentials), `open-source-llm` branch.

You'll need a VM with a GPU for this task.

Please check the [deployment manual](https://github.com/Nebius-Academy/LLMOps-Essentials/blob/main/DEPLOYMENT_MANUAL.md) to learn how to deploy the service on your VM.

**A task for you**.

1. Update the range of available LLMs. Add something new and fashionable
2. Assess the maximal history size that your GPU is able to support
3. (**A quite advanced task**) Try to make the service gather queries into batches before passing them to an LLM. Choose a reasonable maximal batch size. Think about how to populate batches depending on queries' history length
