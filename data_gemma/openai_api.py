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

"""OpenAI LLM Interface."""

import json
import logging
import time
from typing import Any

import requests

from data_gemma import base


class OpenAI(base.LLM):
  """Open AI API."""

  def __init__(
      self,
      model: str,
      api_key: str,
      verbose: bool = True,
      session: requests.Session | None = None,
  ):
    self.key = api_key
    if not session:
      session = requests.Session()
    self.session: requests.Session = session
    self.options = base.Options(verbose=verbose)
    self.model = model

  def query(self, prompt: str) -> base.LLMCall:
    # set the params.
    req_data = {
        'temperature': 0.1,
        'model': self.model,
        'messages': [{
            'role': 'user',
            'content': prompt,
        }],
    }
    # Make API request.
    req = json.dumps(req_data)

    start = time.time()
    self.options.vlog(
        f'... calling OpenAI {self.model} "{prompt[:50].strip()}..."'
    )
    resp = self._call_api(req)
    t = round(time.time() - start, 3)
    ans = ''
    err = ''
    if 'error' in resp:
      err = json.dumps(resp)
      logging.error('%s', err)
      print(err)
    elif (
        'choices' in resp
        and resp['choices']
        and 'message' in resp['choices'][0]
        and 'content' in resp['choices'][0]['message']
    ):
      ans = resp['choices'][0]['message']['content']
    else:
      err = 'Got empty response'
      logging.warning(err)
      print(err)

    return base.LLMCall(prompt=prompt, response=ans, duration_secs=t, error=err)

  def _call_api(self, req_data: str) -> Any:
    header = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {self.key}',
    }
    r = self.session.post(
        'https://api.openai.com/v1/chat/completions',
        data=req_data,
        headers=header,
    )
    return r.json()
