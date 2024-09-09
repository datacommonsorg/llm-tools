# Data Gemma

This repo contains client library code for accessing [DataGemma](paper link), an
open model that helps address the challenges of hallucination by grounding LLMs
in the vast, real-world statistical data of Google's Data Commons.

There are two methodology to achieve this: Retrieval Interleaved Generation
(RIG) and Retrieval Augmented Generation (RAG). More details can be found in the
[paper](link) and [blog post](link).

The finetuned DataGemma models are hosted in HuggingFace
([RIG](https://huggingface.co/google/datagemma-rig-27b-it),
[RAG](https://huggingface.co/google/datagemma-rag-27b-it)) and Kaggle
([RIG](https://www.kaggle.com/models/google/datagemma-rig),
[RAG](https://www.kaggle.com/models/google/datagemma-rag)).

To install the library, run:

```bash
pip install git+https://github.com/datacommonsorg/llm-tools
```

For examples of using this library, see our Colab notebooks for [RIG](https://github.com/datacommonsorg/llm-tools/blob/main/notebooks/data_gemma_rig.ipynb)
and
[RAG](https://github.com/datacommonsorg/llm-tools/blob/main/notebooks/data_gemma_rag.ipynb).

----------
Disclaimer
----------
You're accessing a very early version of DataGemma. It is meant for trusted tester use (primarily for academic and research use) and not yet ready for commercial or general public use. This version was trained on a very small corpus of examples and may exhibit unintended, and at times controversial or inflammatory behavior. Please anticipate errors and limitations as we actively develop this large language model interface.

Your feedback and evaluations are critical to refining DataGemma's performance and will directly contribute to its training process. Known limitations are detailed in the paper, and we encourage you to consult it for a comprehensive understanding of DataGemma's current capabilities.
