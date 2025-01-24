# Copyright 2025 Google LLC
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

"""VertexAI LLM Interface."""

from data_gemma import base
from dataclasses import dataclass
from google.cloud import aiplatform
import time
import logging


@dataclass(kw_only=True)
class VertexAIModelConfig:
  # ID of the GCP project where the model endpoint is deployed
  project_id: str = None
  # The region where the model endpoint is deployed e.g., us-central1
  location: str = None
  # ID of the endpoint to use to make calls to the model
  prediction_endpoint_id: str = None


class VertexAi(base.LLM):
  """VertexAI API."""

  def __init__(
      self,
      project_id: str,
      location: str,
      prediction_endpoint_id: str,
      verbose: bool = True,
  ):
    model_config = VertexAIModelConfig(project_id=project_id,
                                       location=location,
                                       prediction_endpoint_id=prediction_endpoint_id)
    self.prediction_client = _init_client(model_config)
    self.options = base.Options(verbose=verbose)

  def query(self, prompt: str) -> base.LLMCall:
    self.options.vlog(f'... calling Vertex AI API "{prompt[:50].strip()}..."')

    start = time.time()
    instances = [
        { "inputs": f'<bos><start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n'}
    ]

    ans = ''
    err = ''
    try:
      ans = self.prediction_client.predict(instances=instances).predictions[0]
    except Exception as e:
      err = str(e)
      logging.error(err)

    t = round(time.time() - start, 3)

    return base.LLMCall(prompt=prompt, response=ans, duration_secs=t, error=err)


def _init_client(model_config: VertexAIModelConfig):
  aiplatform.init(project=model_config.project_id,
                  location=model_config.location)
  return aiplatform.Endpoint(model_config.prediction_endpoint_id)