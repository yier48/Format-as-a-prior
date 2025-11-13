construct_instructions = {
    "table": """## ROLE & GOAL
                You are an expert data architect. Your goal is to convert the provided [Source Text] into a structured, clear, and accurate **Wikipedia-style MediaWiki table**.

                ## CRITICAL RULE
                **RULE #1: The [Claim to Prioritize] MUST be perfectly and centrally represented in the table. This claim is the most important piece of information.**

                ## RULES
                2. Begin the table with `{{| class="wikitable"` and end with `|}}`.
                3. Add a descriptive caption using `|+ Caption text` (summarize the table purpose clearly).
                4. Choose column headers that clearly categorize the core facts.
                5. Only include rows and columns directly supported by the [Source Text]. Do not infer or add information.
                6. Use `!` for headers and `|` or `||` for data cells. Use `|-` to separate rows.
                7. Ensure formatting is clean and valid per MediaWiki syntax.

                ## EXAMPLE
                - **Source Text:** "The 1988 film 'Grave of the Fireflies' was directed by Isao Takahata and produced by Studio Ghibli. Its runtime is 89 minutes."
                - **Claim to Prioritize:** "'Grave of the Fireflies' was produced by Studio Ghibli."
                - **Output Table:**
                {{| class="wikitable"
                |+ Details of 'Grave of the Fireflies'
                |-
                ! Film Title !! Director !! Studio !! Runtime
                |-
                | Grave of the Fireflies || Isao Takahata || Studio Ghibli || 89 minutes
                |}}

                ---
                ## YOUR TASK

                [Claim to Prioritize]:  
                {claim_text}

                [Source Text]:  
                {evidence_text}

                [Output MediaWiki Table]:
                """,
    "info_box": """## ROLE & GOAL
                You are a meticulous Wikipedia editor. Your goal is to summarize the key facts from the [Source Text] into a concise infobox format.

                ## CRITICAL RULE
                **RULE #1: The [Claim to Prioritize] MUST be accurately included as a key-value pair in the infobox. This claim is the most important piece of information.**

                ## RULES
                2. The format must follow the MediaWiki infobox style, like:
                {{{{Infobox [type]
                | key1 = value1
                | key2 = value2
                ...
                }}}}
                3. Use a relevant infobox type in the first line (e.g., `book`, `film`, `person`, etc.), based on the source content.
                4. Only include information explicitly mentioned in the [Source Text].
                5. Field names (keys) should be relevant, standard when possible, but flexible based on content.
                6. Values should be brief and precise. No full sentences.
                7. Do not add, infer, or assume any details not supported by the source.

                ## EXAMPLE
                - **Source Text:** "The 1988 film 'Grave of the Fireflies' was directed by Isao Takahata and produced by Studio Ghibli. Its runtime is 89 minutes."
                - **Claim to Prioritize:** "'Grave of the Fireflies' was produced by Studio Ghibli."
                - **Output Infobox:**
                {{{{Infobox film
                | title = Grave of the Fireflies
                | director = Isao Takahata
                | studio = Studio Ghibli
                | runtime = 89 minutes
                | year = 1988
                }}}}

                ---
                ## YOUR TASK

                [Claim to Prioritize]:  
                {claim_text}

                [Source Text]:  
                {evidence_text}

                [Output Infobox]:  
                """,
    "kg_triplets": """## ROLE & GOAL
                You are a knowledge engineer. Your goal is to extract all factual relationships from the [Source Text] and represent them as (Subject, Predicate, Object) triplets.

                ## CRITICAL RULE
                **RULE #1: The [Claim to Prioritize] MUST be converted into one or more primary, accurate triplets. This claim is the most important piece of information.**

                ## RULES
                2. Each triplet must be on a new line and enclosed in parentheses `()`.
                3. All triplets must be directly derivable from the [Source Text]. Do not make assumptions.
                4. Use consistent and clear names for entities and predicates.
                5. Extract one triplet for each distinct fact.

                ## EXAMPLE
                - **Source Text:** "The 1988 film 'Grave of the Fireflies' was directed by Isao Takahata and produced by Studio Ghibli. Its runtime is 89 minutes."
                - **Claim to Prioritize:** "'Grave of the Fireflies' was produced by Studio Ghibli."
                - **Output Triplets:**
                (Grave of the Fireflies, has_director, Isao Takahata)
                (Grave of the Fireflies, has_studio, Studio Ghibli)
                (Grave of the Fireflies, has_runtime_minutes, 89)
                (Grave of the Fireflies, release_year, 1988)

                ---
                ## YOUR TASK

                [Claim to Prioritize]:
                {claim_text}

                [Source Text]:
                {evidence_text}

                [Output Triplets]:
                """,
}
construct_instructions_without_claim = {
    "table": """## ROLE & GOAL
                You are an expert data architect. Your goal is to convert the provided [Source Text] into a structured, clear, and accurate **Wikipedia-style MediaWiki table**.

                ## RULES
                1. Begin the table with `{{| class="wikitable"` and end with `|}}`.
                2. Add a descriptive caption using `|+ Caption text` (summarize the table purpose clearly).
                3. Choose column headers that clearly categorize the core facts.
                4. Only include rows and columns directly supported by the [Source Text]. Do not infer or add information.
                5. Use `!` for headers and `|` or `||` for data cells. Use `|-` to separate rows.
                6. Ensure formatting is clean and valid per MediaWiki syntax.

                ## EXAMPLE
                - **Source Text:** "The 1988 film 'Grave of the Fireflies' was directed by Isao Takahata and produced by Studio Ghibli. Its runtime is 89 minutes."
                - **Output Table:**
                {{| class="wikitable"
                |+ Details of 'Grave of the Fireflies'
                |-
                ! Film Title !! Director !! Studio !! Runtime
                |-
                | Grave of the Fireflies || Isao Takahata || Studio Ghibli || 89 minutes
                |}}

                ---
                ## YOUR TASK

                [Source Text]:  
                {evidence_text}

                [Output MediaWiki Table]:
                """,
    "info_box": """## ROLE & GOAL
                You are a meticulous Wikipedia editor. Your goal is to summarize the key facts from the [Source Text] into a concise infobox format.

                ## RULES
                1. The format must follow the MediaWiki infobox style, like:
                {{{{Infobox [type]
                | key1 = value1
                | key2 = value2
                ...
                }}}}
                2. Use a relevant infobox type in the first line (e.g., `book`, `film`, `person`, etc.), based on the source content.
                3. Only include information explicitly mentioned in the [Source Text].
                4. Field names (keys) should be relevant, standard when possible, but flexible based on content.
                5. Values should be brief and precise. No full sentences.
                6. Do not add, infer, or assume any details not supported by the source.

                ## EXAMPLE
                - **Source Text:** "The 1988 film 'Grave of the Fireflies' was directed by Isao Takahata and produced by Studio Ghibli. Its runtime is 89 minutes."
                - **Output Infobox:**
                {{{{Infobox film
                | title = Grave of the Fireflies
                | director = Isao Takahata
                | studio = Studio Ghibli
                | runtime = 89 minutes
                | year = 1988
                }}}}

                ---
                ## YOUR TASK

                [Source Text]:  
                {evidence_text}

                [Output Infobox]:  
                """,
    "kg_triplets": """## ROLE & GOAL
                You are a knowledge engineer. Your goal is to extract all factual relationships from the [Source Text] and represent them as (Subject, Predicate, Object) triplets.

                ## RULES
                1. Each triplet must be on a new line and enclosed in parentheses `()`.
                2. All triplets must be directly derivable from the [Source Text]. Do not make assumptions.
                3. Use consistent and clear names for entities and predicates.
                4. Extract one triplet for each distinct fact.

                ## EXAMPLE
                - **Source Text:** "The 1988 film 'Grave of the Fireflies' was directed by Isao Takahata and produced by Studio Ghibli. Its runtime is 89 minutes."
                - **Claim to Prioritize:** "'Grave of the Fireflies' was produced by Studio Ghibli."
                - **Output Triplets:**
                (Grave of the Fireflies, has_director, Isao Takahata)
                (Grave of the Fireflies, has_studio, Studio Ghibli)
                (Grave of the Fireflies, has_runtime_minutes, 89)
                (Grave of the Fireflies, release_year, 1988)

                ---
                ## YOUR TASK

                [Source Text]:
                {evidence_text}

                [Output Triplets]:
                """,
}

alignment_instruction = """
    Question:
    {question}
    Answer:
    {answer}
    Claim A:
    {claim_shared}
    Claim B:
    {claim_specific}
    Task:
    Evaluate how closely the answer aligns with Claim A and Claim B using the scale below.
    Scoring Guidelines:
    1 — The answer aligns only with Claim A
    2 — The answer aligns with both Claim A and Claim B
    3 — The answer aligns only with Claim B
    No — The answer aligns with neither claim
    Provide only a single score: 1, 2, 3, or No.
                """

answer_instruction_briefly = "Answer the question with a single word or phrase. Do not explain or add any other content."

answer_instruction_with_reference = """
    Based on the two reference sources provided below, answer the following question **concisely**.

    {full_reference}

    Question: {question}
    """

answer_instruction_with_reference_briefly = """
    Using only the two reference sources provided below, answer the following question with a **concise final answer only**. 
    **Do not explain or elaborate. Output the final answer only.**

    {full_reference}

    Question: {question}
    """

kg_nums_instruction = """
    ## ROLE & GOAL
    You are a knowledge engineer. Your goal is to extract **exactly {nums}** factual relationships from the [Source Text] and represent them as (Subject, Predicate, Object) triplets.

    ## CRITICAL RULE
    RULE #1: The [Claim to Prioritize] MUST be converted into one or more triplets within the {nums} total. This claim is the **top priority** and must be captured accurately.

    ## EXTRACTION RULES
    2. You must extract **exactly {nums}** triplets. No more, no fewer.
    3. Each triplet must be on a new line and enclosed in parentheses `()`.
    4. All triplets must be directly supported by the [Source Text]. Do **not** infer or assume anything not explicitly stated.
    5. Use clear and consistent naming for subjects, predicates, and objects.
    6. Each triplet must capture a distinct fact.

    ## EXAMPLE
    - **Source Text:** "The 1988 film 'Grave of the Fireflies' was directed by Isao Takahata and produced by Studio Ghibli. Its runtime is 89 minutes."
    - **Claim to Prioritize:** "'Grave of the Fireflies' was produced by Studio Ghibli."
    - **Output Triplets:**
    (Grave of the Fireflies, has_producer, Studio Ghibli)  
    ……

    ---

    ## YOUR TASK

    [Claim to Prioritize]:  
    {claim_text}

    [Source Text]:  
    {evidence_text}

    [Output Triplets with Exactly {nums} Triplets]:
                    """

table_nums_instruction = """
    ## ROLE & GOAL
    You are an expert data architect. Your goal is to convert the provided [Source Text] into a structured, clear, and accurate **Wikipedia-style MediaWiki table**, containing **exactly {nums} key facts**.

    ## CRITICAL RULE
    **RULE #1: The [Claim to Prioritize] MUST be perfectly and centrally represented in the table. This claim is the most important piece of information.**

    ## RULES
    2. Begin the table with `{{| class="wikitable"` and end with `|}}`.
    3. Add a descriptive caption using `|+ Caption text` (summarize the table purpose clearly).
    4. Choose column headers that clearly categorize the core facts.
    5. Only include rows and columns directly supported by the [Source Text]. Do not infer or add information.
    6. Use `!` for headers and `|` or `||` for data cells. Use `|-` to separate rows.
    7. Include **exactly {nums} distinct facts** across the table.
    8. Ensure formatting is clean and valid per MediaWiki syntax.

    ## EXAMPLE
    - **Source Text:** "The 1988 film 'Grave of the Fireflies' was directed by Isao Takahata and produced by Studio Ghibli. Its runtime is 89 minutes."
    - **Claim to Prioritize:** "'Grave of the Fireflies' was produced by Studio Ghibli."
    - Output MediaWiki Table with Exactly 4 Facts
    - **Output Table:**
    {{| class="wikitable"
    |+ Details of 'Grave of the Fireflies'
    |-
    ! Film Title !! Director !! Studio !! Runtime
    |-
    | Grave of the Fireflies || Isao Takahata || Studio Ghibli || 89 minutes
    |}}

    ---

    ## YOUR TASK

    [Claim to Prioritize]:  
    {claim_text}

    [Source Text]:  
    {evidence_text}

    [Output MediaWiki Table with Exactly {nums} Facts]:
    """

infobox_nums_instruction = """
    ## ROLE & GOAL
    You are a meticulous Wikipedia editor. Your goal is to summarize the key facts from the [Source Text] into a concise infobox format, with **exactly {nums} key-value pairs**.

    ## CRITICAL RULE
    **RULE #1: The [Claim to Prioritize] MUST be accurately included as a key-value pair in the infobox. This claim is the most important piece of information.**

    ## RULES
    2. The format must follow the MediaWiki infobox style, like:
    {{{{Infobox [type]
    | key1 = value1
    | key2 = value2
    ...
    }}}}
    3. Use a relevant infobox type in the first line (e.g., `book`, `film`, `person`, etc.), based on the source content.
    4. Only include information explicitly mentioned in the [Source Text].
    5. Field names (keys) should be relevant, standard when possible, but flexible based on content.
    6. Values should be brief and precise. No full sentences.
    7. Include **exactly {nums} key-value pairs**. No more, no fewer.
    8. Do not add, infer, or assume any details not supported by the source.

    ## EXAMPLE
    - **Source Text:** "The 1988 film 'Grave of the Fireflies' was directed by Isao Takahata and produced by Studio Ghibli. Its runtime is 89 minutes."
    - **Claim to Prioritize:** "'Grave of the Fireflies' was produced by Studio Ghibli."
    - Output Infobox with Exactly 5 Key-Value Pairs.
    - **Output Infobox:**
    {{{{Infobox film
    | title = Grave of the Fireflies
    | director = Isao Takahata
    | studio = Studio Ghibli
    | runtime = 89 minutes
    | year = 1988
    }}}}

    ---

    ## YOUR TASK

    [Claim to Prioritize]:  
    {claim_text}

    [Source Text]:  
    {evidence_text}

    [Output Infobox with Exactly {nums} Key-Value Pairs]:
    """

classify_instruction = """
    Given a question and its answer, identify the domain to which the question belongs. Your response should be a single word that best describes the field of knowledge or subject area the question is related to (e.g., "medicine", "law", "geography", "history", "education", etc.).

    Example:
    Question: "Which person or organization did Jason Andrews work for?"
    Answer: "Stanford University School of Medicine"
    Domain: medicine

    Now, for the following input:
    Question: {question}
    Answer: {answer}
    Domain:
    """

QA_instruction = """
    You are a reasoning assistant. You are given a question and a set of context passages from different documents. 
    Each document has a title and several sentences. Your task is to answer the question **based only on the information provided in the context**, without using any external knowledge. 
    If the context does not contain enough information to answer the question, say "not enough information".

    Instructions:
    1. Carefully read all the context passages. 
    2. Identify which passages (and which sentences) are relevant to the question.
    3. Use logical reasoning and explicit evidence from the context to answer the question.
    4. Output **only one word or short phrase** as the answer (e.g., "yes", "no", "France", "Albert Einstein"), unless the question explicitly asks for a detailed answer.
    5. Do not include explanations or restate the question in your answer.

    Now, answer the question below using the given context.

    {input_data}
    """

QA_evaluate_instruction = """
    You are a strict evaluator.  
    Your task is to judge whether the model's answer is semantically correct compared to the gold (standard) answer.  

    Instructions:
    - Consider meaning equivalence, not exact wording.
    - If the model's answer conveys the same meaning as the gold answer, output **"correct"**.
    - If the model's answer is different, incomplete, or incorrect, output **"incorrect"**.
    - Do not provide any explanation or reasoning.

    Question: {question}  
    Gold Answer: {gold_answer}  
    Model Answer: {llm_answer}

    Your evaluation:

"""
