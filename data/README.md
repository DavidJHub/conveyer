# data/

Place the conversations parquet here (not versioned — see `.gitignore`):

    data/conversations.parquet

Expected schema (grain = one user↔LLM turn; full details in
[`docs/DATA_DICTIONARY.md`](../docs/DATA_DICTIONARY.md) §0):

| column | description |
|---|---|
| `session_id` / `message_id` / `user_id` | identifiers |
| `prompt_datetime` | prompt timestamp |
| `question` / `answer` | user prompt / model response |
| `a_links_source` | list of links surfaced in the response |
| `ai_click` | list of links clicked directly from the AI response |
| `next_10_urls` | list of `{request_time, requested_site}` navigation events |

When the file is absent, every entry point (`python -m conveyer.pipeline`, the
notebooks, the tests) falls back to a **synthetic dataset with ground truth**,
so the whole pipeline runs and validates itself before real data lands.
