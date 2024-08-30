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

"""List of all prompts used in Data Gemma."""

# TODO: Unify this with RIG_IN_CONTEXT_PROMPT.  No reason why this should be
# different, and include the "question"
RIO_IN_CONTEXT_PROMPT = """
Your task is to annotate every statistic in the given text with a `__DC__`
query that can retrieve the statistic.  The query should be about metrics
on topics like demographics, economy, education, health, and so on that are
associated with geographical places (like USA, California, Miami, etc.).

Concretely, every occurrence of a statistical value for a metric in a place
should be replaced with `[__DC__("query") --> "stat"]`, where "query"
must include a metric, a place name and optional date. And "stat" is the
statistical value that originally occurred in the text.  Do not annotate
values that are dates ("founded in 1760") and ranks ("10th largest by area").

The `__DC__()` calls MUST be in place of the statistical value in the text.
And DO NOT modify sentences that have no statistical data.

Below is an example of an INPUT and the corresponding annotated OUTPUT.

INPUT:

Question:- Tell me one statistic about California, San Francisco, Alabama and the US.
Answer:-
California is 1st as the nation's most populous state, with about 39 million people in 2020.
In San Francisco, the diabetes rate is 9.2 cases per 10000 people.
San Francisco and the surrounding San Francisco Bay Area are a global center of economic activity and the arts and sciences.
In 1861, Alabama seceded from the United States to become part of the Confederate States of America.
As of 2022, the United States receives approximately 81% of its energy from fossil fuel and the largest source of the country's energy came from petroleum (35.8%), followed by natural gas (33.4%) and renewable sources (13.3%).

OUTPUT:

Question:- Tell me one statistic about California, San Francisco, Alabama and the US.
Answer:-
California is 1st as the nation's most populous state, with about [__DC__("what was the population of california in 2020?") --> "39 million"] people.
In San Francisco, the diabetes rate is [__DC__("what is the prevalence of diabetes in San Francisco?") --> "9.2 cases per 10000 people"].
San Francisco and the surrounding San Francisco Bay Area are a global center of economic activity and the arts and sciences.
In 1861, Alabama seceded from the United States to become part of the Confederate States of America.
As of 2022, the United States receives approximately [__DC__("what percentage of energy in the United States comes from fossil fuels in 2022?") --> "81%"] of its energy from fossil fuel and the largest source of the country's energy came from petroleum ([__DC__("what percentage of energy in the United States comes from petroleum in 2022?") --> "35.8%"]), followed by natural gas ([__DC__("what percentage of energy in the United States comes from natural gas in 2022?") --> "33.4%"]) and renewable sources ([__DC__("what percentage of energy in the United States comes from renewable sources in 2022?") --> "13.3%"]).

INPUT:

Question:- {question}
Answer:-
{answer}

OUTPUT:

"""


RIG_IN_CONTEXT_PROMPT = """
Your task is to annotate every statistic in the given text with a `__DC__`
query that can retrieve the statistic.  The query should be about metrics
on topics like demographics, economy, education, health, and so on that are
associated with geographical places (like USA, California, Miami, etc.).

Concretely, every occurrence of a statistical value for a metric in a place
should be replaced with `[__DC__("query") --> "stat"]`, where "query"
must include a metric, a place name and optional date. And "stat" is the
statistical value that originally occurred in the text.  Do not annotate
values that are dates ("founded in 1760") and ranks ("10th largest by area").

The `__DC__()` calls MUST be in place of the statistical value in the text.
And DO NOT modify sentences that have no statistical data.

Below is an example of an INPUT and the corresponding annotated OUTPUT.

INPUT:

California is 1st as the nation's most populous state, with about 39 million people in 2020.
In San Francisco, the diabetes rate is 9.2 cases per 10000 people.
San Francisco and the surrounding San Francisco Bay Area are a global center of economic activity and the arts and sciences.
In 1861, Alabama seceded from the United States to become part of the Confederate States of America.
As of 2022, the United States receives approximately 81% of its energy from fossil fuel and the largest source of the country's energy came from petroleum (35.8%), followed by natural gas (33.4%) and renewable sources (13.3%).

OUTPUT:

California is 1st as the nation's most populous state, with about [__DC__("what was the population of california in 2020?") --> "39 million"] people.
In San Francisco, the diabetes rate is [__DC__("what is the prevalence of diabetes in San Francisco?") --> "9.2 cases per 10000 people"].
San Francisco and the surrounding San Francisco Bay Area are a global center of economic activity and the arts and sciences.
In 1861, Alabama seceded from the United States to become part of the Confederate States of America.
As of 2022, the United States receives approximately [__DC__("what percentage of energy in the United States comes from fossil fuels in 2022?") --> "81%"] of its energy from fossil fuel and the largest source of the country's energy came from petroleum ([__DC__("what percentage of energy in the United States comes from petroleum in 2022?") --> "35.8%"]), followed by natural gas ([__DC__("what percentage of energy in the United States comes from natural gas in 2022?") --> "33.4%"]) and renewable sources ([__DC__("what percentage of energy in the United States comes from renewable sources in 2022?") --> "13.3%"]).

INPUT:

{text}

OUTPUT:

"""


RAG_IN_CONTEXT_PROMPT = """
Given a QUERY below, your task is to come up with a maximum of 25
STATISTICAL QUESTIONS that help in answering QUERY.

Here are the only forms of STATISTICAL QUESTIONS you can generate:

1. "What is $METRIC in $PLACE?"
2. "What is $METRIC in $PLACE $PLACE_TYPE?"
3. "How has $METRIC changed over time in $PLACE $PLACE_TYPE?"

where:
- $METRIC should a publicly accessible metric on societal topics around
  demographics, economy, health, education, environment, etc.  Examples are
  unemployment rate, life expectancy, etc.
- $PLACE is the name of a place like California, World, Chennai, etc.
- $PLACE_TYPE is an immediate child type within $PLACE, like counties, states,
  districts, etc.

Your response should only include the questions, one per line without any
numbering or bullet!  If you cannot come up with statistical questions to ask,
return an empty response.

NOTE:  Do not repeat questions.  Limit the number of questions to 25.

If QUERY asks about  multiple concepts (e.g., income and diseases), make sure
the questions cover all the concepts.

[Start of Examples]

QUERY: Which grades in the middle school have the lowest enrollment in Palo Alto?
STATISTICAL QUESTIONS:
What is the number of students enrolled in Grade 6 in Palo Alto schools?
What is the number of students enrolled in Grade 7 in Palo Alto schools?
What is the number of students enrolled in Grade 8 in Palo Alto schools?

QUERY: Which industries have grown the most in California?
STATISTICAL QUESTIONS:
How have jobs in agriculture changed over time in California?
How has GDP of agriculture sector changed over time in California?
How have jobs in information and technology changed over time in California?
How has GDP of information and technology sector changed over time in California?
How have jobs in the government changed over time in California?
How has GDP of the government sector changed over time in California?
How have jobs in healthcare changed over time in California?
How has GDP of healthcare sector changed over time in California?
How have jobs in entertainment changed over time in California?
How has GDP of entertainment sector changed over time in California?
How have jobs in retail trade changed over time in California?
How has GDP of retail trade sector changed over time in California?
How have jobs in manufacturing changed over time in California?
How has GDP of manufacturing sector changed over time in California?
How have jobs in education services changed over time in California?
How has GDP of education services sector changed over time in California?

QUERY: Which state in the US has the most asian population?
STATISTICAL QUESTIONS:
What is the number of asian people in US states?

QUERY: Do specific health conditions affect the richer California counties?
STATISTICAL QUESTIONS:
What is the median income among California counties?
What is the median house price among California counties?
What is the prevalence of obesity in California counties?
What is the prevalence of diabetes in California counties?
What is the prevalence of heart disease in California counties?
What is the prevalence of arthritis in California counties?
What is the prevalence of asthma in California counties?
What is the prevalence of chronic kidney disease in California counties?
What is the prevalence of chronic obstructive pulmonary disease in California counties?
What is the prevalence of coronary heart disease in California counties?
What is the prevalence of high blood pressure in California counties?
What is the prevalence of high cholesterol in California counties?
What is the prevalence of stroke in California counties?
What is the prevalence of poor mental health in California counties?
What is the prevalence of poor physical health in California counties?


[End of Examples]

QUERY: {sentence}
STATISTICAL QUESTIONS:
"""


RAG_IN_CONTEXT_PROMPT_WITH_VARS = """
Given a 'Query' below, your task is to come up with a maximum of 25
'Statistical Questions' that relate to 'Query'.

Here are the only forms of 'Statistical Questions' you can generate:

1. What is $METRIC in $PLACE?
2. What is $METRIC in $PLACE $PLACE_TYPE?
3. How has $METRIC changed over time in $PLACE $PLACE_TYPE?

Where:
- $METRIC should only be from the 'Metrics List' given below.
- $PLACE is the name of a place like California, World, Chennai, etc.
- $PLACE_TYPE is first-level child type within $PLACE, like counties or
  districts if $PLACE is a state, states if $PLACE is a country, etc.

Your response should only include the questions, one per line, without any
numbering or bullets! If you cannot come up with 'Statistical Questions' only
using the 'Metrics List' below, return an empty response.

NOTE:  Do not repeat questions.  Limit the number of questions to 25 and
order the questions from most relevant to least relevant.

If "Query" asks about  multiple concepts (e.g., income and diseases), make sure
the questions cover all the concepts.

[Start of Examples]

Query: Tell me about life expectancy.
Statistical Questions:
What is the people life expectancy in the world?
How has people life expectancy changed over time in the world countries?

Query: Which state in the US has the most asian population?
Statistical Questions:
What is the number of asian people in US states?
How has the number of asian people changed over time in US states?

Query: Which grades in the middle school have the lowest enrollment in Palo Alto?
Statistical Questions:
What is the number of students enrolled in Grade 6 in Palo Alto schools?
What is the number of students enrolled in Grade 7 in Palo Alto schools?
What is the number of students enrolled in Grade 8 in Palo Alto schools?

QUERY: Do specific health conditions affect the richer California counties?
STATISTICAL QUESTIONS:
What is the median income among California counties?
What is the median house price among California counties?
What is the prevalence of obesity in California counties?
What is the prevalence of diabetes in California counties?
What is the prevalence of heart disease in California counties?
What is the prevalence of arthritis in California counties?
What is the prevalence of asthma in California counties?
What is the prevalence of chronic kidney disease in California counties?
What is the prevalence of chronic obstructive pulmonary disease in California counties?
What is the prevalence of coronary heart disease in California counties?
What is the prevalence of high blood pressure in California counties?
What is the prevalence of high cholesterol in California counties?
What is the prevalence of stroke in California counties?
What is the prevalence of poor mental health in California counties?
What is the prevalence of poor physical health in California counties?

[End of Examples]


[Start of Metrics List]

Here is a list of possible METRIC values:

```
{metrics_list}
```

[End of Metrics List]


Query: {sentence}
Statistical Questions:
"""


RAG_FINE_TUNED_PROMPT = """"
Your role is that of a Question Generator.  Given Query below, come up with a
maximum of 25 Statistical Questions that help in answering Query.

These are the only forms of Statistical Questions you can generate:
1. What is $METRIC in $PLACE?
2. What is $METRIC in $PLACE $PLACE_TYPE?
3. How has $METRIC changed over time in $PLACE $PLACE_TYPE?

where,
- $METRIC should a metric on societal topics like demographics, economy, health,
  education, environment, etc.  Examples are unemployment rate and
  life expectancy.
- $PLACE is the name of a place like California, World, Chennai, etc.
- $PLACE_TYPE is an immediate child type within $PLACE, like counties, states,
  districts, etc.

Your response should only have questions, one per line, without any numbering
or bullet.

If you cannot come up with Statistical Questions to ask for a Query, return an
empty response.

Query: {sentence}
Statistical Questions:
"""


RAG_FINAL_ANSWER_PROMPT = """
Using statistics from the tables below, respond to the query: "{sentence}"

In your response, when using statistics from a table, please cite the table
by its ID, enclosed in square brackets. For example, for one table reference
`[Table 1]`, and for multiple tables `[Table 1], [Table 2] and [Table 3]`.

If necessary to answer the query, perform simple calculations on the statistics,
like adding or subtracting statistics, computing growth rates from statistics
over time, etc.

If you are unable to answer the query based on the provided tables, start your
response with `[NO ANSWER]` as the first line.

```
{table_str}
```

So now, using statistics from the tables above, respond to the query: "{sentence}"
"""


DC_QA_VALIDATION = """
You will be provided with a list of up to 20 question-answer pairs, each
identified by an ID like [[QA1]].  You must return each ID whose answer is
relevant to its question, one per line. If none of the answers are relevant,
return `[[EMPTY]]`.

Here is an example INPUT and OUTPUT:

## INPUT ##
[[QA1]]:
  Question: "What is the average education spending per pupil in New York?"
  Answer: "% Govt Expenditure on Education in United States"
[[QA2]]
  Question: "What is the Gini coefficient in Chile?"
  Answer: "Gini Index of Economic Activity of a Population in Chile"
[[QA3]]
  Question: "How many people work in health care jobs in Nevada?"
  Answer: "Population of Health Care Workers in Nevada"

## OUTPUT ##
[[QA2]]
[[QA3]]


## INPUT ##
{input}

## OUTPUT ##
"""


LLM_JUDGE_PROMPT = """
[System]

Please act as an impartial judge and evaluate the quality of the response
provided by an AI Assistant that annotates statistical numbers with questions.

In the text, you will find patterns of the form: [__DC__("QUERY") --> "ANS"].
Where, `ANS` is a statistical value, and `QUERY` is a query that can be answered
with `ANS`.

Your evaluation should consider whether the `__DC__` annotations follow these
constraints:

(C1) `QUERY` can refer to a very wide variety of measures related to demographics,
     economy, education, health, etc. However, it should accurately describe the
     statistic involved. `QUERY` must include a place name of a city, state,
     country, continent, etc, or words that represent the world (like global).

(C2) `ANS` must not be empty or have the the word "stat".  `ANS` must have a
     numeric value, may include percentages, and may additionally have
     non-numeric letters for units, currency symbols, etc.

Provide a classification of the answer quality like "[[<value>]]",
where the <value> can be:
- GOOD: The annotations in the answer adhere to the above rules.
- BAD: If some annotations do not follow the above rules.

NOTE: You do not need to judge the correctness of the `ANS` value. Duplicate
annotations are fine.

For example, if the answer is GOOD, the first line of response should be
"[[GOOD]]".

Then, list only the bad `[__DC__("QUERY") --> "ANS"]` values, one per line,
and concisely point out what is wrong.

You don't have to provide a revised version of the answer.

[Start of Assistant's Answer]

{answer}

[End of Assistant's Answer]
"""
