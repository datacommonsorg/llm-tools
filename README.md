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
