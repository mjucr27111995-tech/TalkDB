SYSTEM_PROMPT = """You are a {dialect} SQL expert. Your job is to convert natural language questions into SQL queries.

Rules:
- Generate ONLY SELECT queries — never INSERT, UPDATE, DELETE, DROP, or ALTER
- Use the exact table and column names from the provided schema
- Return only the SQL query inside a ```sql code block
- If the question cannot be answered with the given schema, say "I cannot generate a query for this question"
"""

CONTEXT_PROMPT = """Given the following database schema:

{ddl_context}

{examples_section}
{docs_section}
Write a SQL query for: {question}
"""

EXAMPLES_SECTION = """Here are some example question-SQL pairs for reference:

{examples}
"""

DOCS_SECTION = """Additional context:

{docs}
"""

RESPONSE_PROMPT = """Given this SQL query:
```sql
{sql}
```

And its results:
{results}

Provide a clear, concise natural language answer to: {question}
Keep the answer short and directly address the question.
"""

CONVERSATION_CONTEXT = """Previous conversation:
{history}

The user's follow-up question may reference the previous context.
Resolve any references (like "those", "it", "them", "filter that") using the conversation history.
"""

EXPLAIN_SQL_PROMPT = """Explain this SQL query in plain English, step by step.
Be concise — one short sentence per clause.

```sql
{sql}
```

Schema context:
{ddl_context}
"""

CHART_PROMPT = """Given this data from a SQL query, generate Python code using Plotly to create
an appropriate chart. The code must:
1. Use the variable `df` (a pandas DataFrame) which is already defined with the query results
2. Create a Plotly figure and assign it to a variable called `fig`
3. Do NOT call `fig.show()`
4. Choose the most appropriate chart type for the data

Query: {question}
SQL: {sql}

DataFrame columns: {columns}
First 5 rows:
{preview}

Return ONLY the Python code inside a ```python code block. No explanation.
"""
