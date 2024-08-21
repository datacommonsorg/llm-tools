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

"""Data Commons."""

import concurrent.futures
import csv
import io
from typing import Any, Callable

import requests

from data_gemma import base
from data_gemma import utils


_BASE_URL = 'https://{env}.datacommons.org/nodejs/query'

# Do not allow topics, use higher threshold (0.8).
_POINT_PARAMS = 'allCharts=1&mode=toolformer_rig&idx=base_uae_mem'

# Allow topics, use lower threshold (0.7)
_TABLE_PARAMS = 'mode=toolformer_rag&client=table&idx=base_uae_mem'


class DataCommons:
  """Data Commons."""

  def __init__(
      self,
      api_key: str,
      verbose: bool = True,
      num_threads: int = 10,
      env: str = 'dev',
      session: requests.Session | None = None,
  ):
    self.options = base.Options(verbose=verbose)
    self.num_threads = num_threads
    self.env = env
    self.api_key = api_key
    if not session:
      session = requests.Session()
    self.session = session

  def point(self, query: str) -> base.DataCommonsCall:
    """Calls Data Commons API."""

    self.options.vlog(f'... calling DC with "{query}"')
    response = self._call_api(query, _POINT_PARAMS)
    # Get the first LINE chart.
    chart = None
    for c in response.get('charts', []):
      ctype = c.get('type')
      if ctype == 'LINE' or ctype == 'HIGHLIGHT':
        chart = c
        break
    if not chart:
      return base.DataCommonsCall(query=query)

    v = str(chart.get('highlight', {}).get('value', ''))
    v = utils.round_float(v)
    if not v:
      return base.DataCommonsCall(query=query)

    u = chart.get('unit', '')
    d = chart.get('highlight', {}).get('date')
    s = _src(chart)
    t = chart.get('title', '')

    svm = response.get('debug', {}).get('debug', {}).get('sv_matching', {})
    score = svm.get('CosineScore', [-1])[0]
    var = svm.get('SV', [''])[0]
    url = chart.get('dcUrl', '')
    return base.DataCommonsCall(
        query=query,
        val=v,
        unit=u,
        title=t,
        date=d,
        src=s,
        url=url,
        var=var,
        score=score,
    )

  def table(self, query: str) -> base.DataCommonsCall:
    """Calls Data Commons API."""

    self.options.vlog(f'... calling DC for table with "{query}"')
    response = self._call_api(query, _TABLE_PARAMS)
    # Get the first chart.
    charts = response.get('charts')
    if not charts:
      return base.DataCommonsCall(query=query)
    chart = charts[0]

    data_csv = chart.get('data_csv', '')
    rows = list(csv.reader(io.StringIO(data_csv)))
    if not data_csv or not rows:
      return base.DataCommonsCall(query=query)

    u = chart.get('unit', '')
    s = _src(chart)
    t = chart.get('title', '')

    parts = []
    parts.append(' | '.join(rows[0]))
    parts.append('-' * len(parts[-1]))
    for row in rows[1:]:
      row = [utils.round_float(v) for v in row]
      parts.append(' | '.join(row))
    parts.append('\n')
    table_str = '\n'.join(parts)

    svm = response.get('debug', {}).get('debug', {}).get('sv_matching', {})
    score = svm.get('CosineScore', [-1])[0]
    var = svm.get('SV', [''])[0]
    url = chart.get('dcUrl', '')
    return base.DataCommonsCall(
        query=query,
        unit=u,
        title=t,
        src=s,
        table=table_str,
        url=url,
        var=var,
        score=score,
    )

  def calln(
      self, queries: list[str], func: Callable[[str], base.DataCommonsCall]
  ) -> dict[str, base.DataCommonsCall]:
    """Calls Data Commons API in parallel if needed."""

    if self.num_threads == 1:
      results = [func(q) for q in queries]
    else:
      # TODO: Check why this ~breaks in Colab Borg runtime
      with concurrent.futures.ThreadPoolExecutor(self.num_threads) as executor:
        futures = [executor.submit(func, query) for query in queries]
        results = [f.result() for f in futures]

    q2resp: dict[str, base.DataCommonsCall] = {}
    for i, (q, r) in enumerate(zip(queries, results)):
      r.id = i + 1
      q2resp[q] = r
    return q2resp

  def _call_api(self, query: str, extra_params: str) -> Any:
    query = query.strip().replace(' ', '+')
    url = _BASE_URL.format(env=self.env) + f'?&q={query}&{extra_params}'
    if self.api_key:
      url = f'{url}&apikey={self.api_key}'
    # print(f'DC: Calling {url}')
    return self.session.get(url).json()


def _src(chart: dict[str, Any]) -> str:
  srcs = chart.get('srcs', [{}])
  if not srcs:
    return ''
  return srcs[0].get('name', '')
