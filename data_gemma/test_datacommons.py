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
"""Integration tests for datacommons.py that make real API calls."""

import os
import unittest

from data_gemma import datacommons

class DataCommonsIntegrationTest(unittest.TestCase):

  def test_call_api_real(self):
    """Tests a real call to the Data Commons API to check header handling."""
    api_key = os.environ.get('DC_API_KEY')
    if not api_key:
      self.skipTest("DC_API_KEY environment variable not set. Skipping integration test.")

    dc = datacommons.DataCommons(api_key=api_key)
    
    # A simple query that should return a valid response.
    query = "population of usa"
    params = "mode=toolformer_rag&client=table&idx=base_uae_mem"
    
    try:
      response = dc._call_api(query, params)
      print("response! ", response)
      # Check for a successful response (a dictionary with expected keys)
      self.assertIsInstance(response, dict)
      self.assertIn('charts', response)
      print("Successfully received a valid JSON response from the API.")
    except Exception as e:
      self.fail(f"_call_api raised an exception with a real API call: {e}")

if __name__ == '__main__':
  unittest.main()
