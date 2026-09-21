You should only provide a concise summary of the key content from the user's text. **The summary you generate will be vectorized and reused for retrieval, so it must never include repetitive preambles or identical introductory phrases across different contexts.** Based on the given context, **summarize the core content into one complete paragraph** and output it **in Korean.**

### [Output Rules]
1. If the context only repeats titles or lacks meaningful content that makes summarization impossible, return only the word "NO_SUMMARY" without any other text.
2. Begin immediately with specific content details - NEVER start with generic phrases like "이 컨텍스트는", "이 문서는", "제시된 내용은", "텍스트는", "자료는", "내용은", or any similar meta-descriptive language.
3. Start directly with the core subject matter, facts, or findings without describing the document itself.
4. Avoid using the same opening phrases across different summaries to prevent vector similarity degradation.
5. Focus on extracting and condensing the essential information while preserving key concepts and relationships for retrieval purposes.
6. Ensure all critical keywords and technical terms from the original text are retained in the summary to maintain searchability.
7. Verify that the summary captures the main topics and domain-specific terminology that users might search for.
8. Output only the summary paragraph or "NO_SUMMARY" - no additional text allowed.

[Context]
{text}