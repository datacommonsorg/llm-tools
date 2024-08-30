# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""HF Pipeline API based LLM Interface.

For example usage, see: https://huggingface.co/google/gemma-2-27b
"""

import logging
import time
from typing import Any

from data_gemma import base

MAX_NEW_TOKENS = 1024


class HFPipeline(base.LLM):
  """HuggingFace Pipeline API."""

  def __init__(
      self,
      pipeline: Any,
      verbose: bool = True,
  ):
    self.pipeline = pipeline
    self.options = base.Options(verbose=verbose)

  def query(self, prompt: str) -> base.LLMCall:
    self.options.vlog(f'... calling HF Pipeline API "{prompt[:50].strip()}..."')

    start = time.time()
    outputs = self.pipeline(prompt, max_new_tokens=MAX_NEW_TOKENS)
    t = round(time.time() - start, 3)

    ans = ''
    err = ''
    if not outputs:
      err = 'Empty outputs from pipeline() call!'
    elif 'generated_text' not in outputs[0]:
      err = 'generated_text not found in outputs[0]!'
    else:
      ans = outputs[0]['generated_text']

    if err:
      logging.warning(err)
      print(f'WARNING: {err}')

    return base.LLMCall(prompt=prompt, response=ans, duration_secs=t, error=err)


class HFBasic(base.LLM):
  """HuggingFace Model / Tokenizer API.

  Note: Model is assumed to be loaded on a GPU.
  """

  def __init__(
      self,
      model: Any,
      tokenizer: Any,
      verbose: bool = True,
  ):
    self.model = model
    self.tokenizer = tokenizer
    self.options = base.Options(verbose=verbose)

  def query(self, prompt: str) -> base.LLMCall:
    self.options.vlog(f'... calling HF Pipeline API "{prompt[:50].strip()}..."')

    start = time.time()
    input_ids = self.tokenizer(prompt, return_tensors='pt').to('cuda')
    outputs = self.model.generate(**input_ids, max_new_tokens=4096)

    ans = ''
    err = ''
    try:
      ans = self.tokenizer.decode(outputs[0])
    except Exception as e:
      err = str(e)
      logging.warning(err)
      print(f'WARNING: {err}')

    t = round(time.time() - start, 3)

    return base.LLMCall(prompt=prompt, response=ans, duration_secs=t, error=err)
