import json
import re
from utils import extract_json_from_end


# Base prompt: Alice reviews Bob's state using the problem description and both states.
CRITIC_PROMPT_BASE0 = """You are Alice, an expert in operations research and optimization. You are collaborating with another expert, Bob, to solve a complex problem. Your proposed mathematical formulations differ, but there is only one correct solution.

Your task is to conduct a rigorous peer review of Bob's formulation. By comparing it against your own and the problem's fundamental requirements, your goal is to identify any discrepancies and provide constructive feedback to help converge on the correct model.

**Problem Description:**
{description}

**Your Formulation:**
-----
{state_alice}
-----

**Bob's Formulation:**
-----
{state_bob}
-----

**Instructions:**

Critically analyze Bob's formulation, focusing on its mathematical correctness and completeness. Your feedback must be structured within a JSON object, addressing the following four components: `parameters`, `variables`, `constraints`, and `objective`.

For each component, provide specific, actionable comments. The following content must be included in your analysis:
1.  **The Issue:** What is incorrect or suboptimal?
2.  **The Justification:** Why is it an issue?
3.  **The Recommendation:** How should it be corrected?

Maintain a professional, analytical tone. Focus exclusively on the mathematical formulation; do not provide or discuss implementation code. If you truly believe that one component in Bob's formulation is entirely correct, leave its corresponding value as an empty string.

**Output Format:**
=====
{{
    "parameters": "YOUR COMMENTS",
    "variables": "YOUR COMMENTS",
    "constraints": "YOUR COMMENTS",
    "objective": "YOUR COMMENTS"
}}
=====
"""

# Base prompt: Alice revises her state using the description and Bob's comments.
REFORMULATION_PROMPT_BASE0 = """You are Alice, an expert in operations research and optimization. After reviewing each other's work, your collaborator, Bob, has provided feedback on your initial formulation.

Your objective is to produce the definitive, correct mathematical formulation for the problem. This requires you to critically evaluate Bob's comments and, based on your expert judgment, refine your model.

**Problem Description:**
{description}

**Bob's Comments on Your Formulation:**
{comments_bob}

**Your Previous Formulation:**
-----
{state_alice}
-----

**Instructions:**
1.  **Evaluate Feedback:** Carefully analyze each comment from Bob. Determine if the proposed changes are valid and lead to a more accurate or robust model.
2.  **Synthesize and Refine:** Integrate any valid feedback into your formulation. You may modify any part of the model (parameters, variables, constraints, objective) to achieve correctness.
3.  **Produce the Final Model:** Output the complete, final version of the mathematical formulation. If you determine that Bob's feedback is not valid and your original formulation stands as correct, resubmit it without changes.

**Output Format (MUST follow strictly):**
Remain the same as before. Just make sure that there's only one json object in the output, and it should be the reformulated state. Maintain the same structure as your previous formulation.

"""

# Revision-history variant: Alice also sees her earlier comments and Bob's replies.
# `history_revision` is a JSON object with the following structure:
# {
#     round_1: {
#         "alice_comment_on_bobs_state": "...",
#         "bob_reply_after_reformulation": "..."
#     }
# }
CRITIC_PROMPT_BASE1 = """You are Alice, an expert in operations research and optimization. You are collaborating with another expert, Bob, to solve a complex problem. Your proposed mathematical formulations differ, but there is only one correct solution.

Your objective is to conduct a rigorous peer review of Bob's formulation. By comparing it against your own and the problem's fundamental requirements, your goal is to identify any discrepancies and provide constructive feedback to help converge on the correct model.

**Problem Description:**
{description}

**Your Formulation:**
-----
{state_alice}
-----
**Bob's Formulation:**
-----
{state_bob}
-----
**Previous Interactions:**
{revision_history}

**Instructions:**

Critically analyze Bob's formulation, focusing on its mathematical correctness and completeness. Your feedback must be structured within a JSON object, addressing the following four components: `parameters`, `variables`, `constraints`, and `objective`.

For each component, provide specific, actionable comments. The following content must be included in your analysis. But also make your comments brief. Each following content must be finished within 1 sentence.
1.  **The Issue:** What is incorrect or suboptimal?
2.  **The Justification:** Why is it an issue?
3.  **The Recommendation:** How should it be corrected?

Focus exclusively on the mathematical formulation; do not provide or discuss implementation code. If you truly believe that one component in Bob's formulation is entirely correct, leave its corresponding value as an empty string.

**Output Format:**
=====
{{
    "parameters": "YOUR COMMENTS",
    "variables": "YOUR COMMENTS",
    "constraints": "YOUR COMMENTS",
    "objective": "YOUR COMMENTS"
}}
=====
"""

# Paired with CRITIC_PROMPT_BASE1, this variant adds a short reply after each
# reformulation summarizing which comments were accepted or rejected.
# The output contains two parts: a reformulated-state JSON object and a reply
# wrapped in <reply> ... </reply>.
REFORMULATION_PROMPT_BASE1 = """You are Alice, an expert in operations research and optimization. After reviewing each other's work, your collaborator, Bob, has provided feedback on your initial formulation.

Your objective is to produce the definitive, correct mathematical formulation for the problem. This requires you to critically evaluate Bob's comments and, based on your expert judgment, refine your model. 

Remember to see the revision history of your mutual discussions. The discussion continue until both of you reach a consensus. But don't be fully guided by Bob's comments. You should make your own judgment on whether to accept or reject each comment.

**Problem Description:**
{description}
**Bob's Comments on Your Formulation:**
{comments_bob}
**Your Previous Formulation:**
-----
{state_alice}
-----
**Previous Interactions:**
{revision_history}
**Instructions:**
1.  **Evaluate Feedback:** Carefully analyze each comment from Bob. Determine if the proposed changes are valid and lead to a more accurate or robust model.
2.  **Synthesize and Refine:** Integrate any valid feedback into your formulation. You may modify any part of the model (parameters, variables, constraints, objective) to achieve correctness.
3.  **Produce the Final Model:** Output the complete, final version of the mathematical formulation. If you determine that Bob's feedback is not valid and your original formulation stands as correct, resubmit it without changes.
4.  **Provide a Brief Reply:** Accompany your reformulated state with a concise reply to Bob. In this reply, summarize which of his comments you found valid and incorporated into your model, and which you did not, along with brief justifications. The brief reply should be enclosed within <reply> ... </reply> tags.

**Output Format (MUST follow strictly):**
Remain the formulation json the same as before. Give both the reformulated state and the reply. Make sure that there's only one json object in the output, and it should be the reformulated state. Maintain the same structure as your previous formulation. The reply should be in the format:
<reply>
YOUR REPLY TEXT
</reply>
"""

CRITIC_PROMPT_BASE2 = """You are Alice, an expert in operations research and optimization. You are collaborating with another expert, Bob, to solve a complex problem. Your proposed mathematical formulations differ, but there is only one correct solution.

Your objective is to conduct a rigorous peer review of Bob's formulation. By comparing it against your own and the problem's fundamental requirements, your goal is to find out any mistakes that Bob made and provide constructive feedback to help converge on the correct model.

You and Bob have been iterating on your formulations, and until now you two have not yet reach a consensus. For a better comment, you should also take as reference your previous comments on Bob's formulation and Bob's replies after each reformulation.

**Problem Description:**
{description}

**Your Previous Formulation:**
-----
{state_alice}
-----
**Bob's Formulation:**
-----
{state_bob}
-----
**Previous Interactions on Bob's formulation**
{revision_history}

**Instructions:**

Critically analyze Bob's formulation, focusing on its mathematical correctness and completeness. Your feedback must be structured within a JSON object, addressing the following four components: `parameters`, `variables`, `constraints`, and `objective`.

For each component, provide specific, actionable comments. The following content must be included in your analysis. But also make your comments brief. Each following content must be finished within 1 sentence.
1.  **The Issue:** What is incorrect or suboptimal?
2.  **The Justification:** Why is it an issue?
3.  **The Recommendation:** How should it be corrected?

Focus exclusively on the mathematical formulation; do not provide or discuss implementation code. If you truly believe that one component in Bob's formulation is entirely correct, leave its corresponding value as an empty string.

**Output Format:**
=====
{{
    "parameters": "YOUR COMMENTS",
    "variables": "YOUR COMMENTS",
    "constraints": "YOUR COMMENTS",
    "objective": "YOUR COMMENTS"
}}
=====
"""


REFORMULATION_PROMPT_BASE2 = """You are Alice, an expert in operations research and optimization. After reviewing each other's work, your collaborator, Bob, has provided feedback on your initial formulation.

Your objective is to produce the definitive, correct mathematical formulation for the problem. This requires you to critically evaluate Bob's comments and, based on your expert judgment, refine your model. Remember, Bob may also make mistakes. If you are confident that your original formulation is correct, you may choose to keep it unchanged.

You and Bob have been iterating on your formulations, and until now you two have not yet reach a consensus. For a better reformulation, you should also take as reference Bob's previous comments on your formulation and your replies after each reformulation.  

**Problem Description:**
{description}

**Bob's Comments on Your Formulation:**
{comments_bob}

**Your Previous Formulation:**
-----
{state_alice}
-----

**Previous Interactions on Your Formulation**
{revision_history}

**Instructions:**
1.  **Evaluate Feedback:** Carefully analyze each comment from Bob. Determine if the proposed changes are valid and lead to a more accurate or robust model.
2.  **Synthesize and Refine:** Integrate any valid feedback into your formulation. You may modify any part of the model (parameters, variables, constraints, objective) to achieve correctness.
3.  **Produce the Final Model:** Output the complete, final version of the mathematical formulation. If you determine that Bob's feedback is not valid and your original formulation stands as correct, resubmit it without changes.
4.  **Provide a short Reply:** In this reply, summarize which of his comments you found valid and incorporated into your model, and which you did not, along with brief justifications. The reply should be enclosed within <reply> ... </reply> tags.

**Output Format (MUST follow strictly):**
Remain the formulation json the same as before. Give both the reformulated state and the reply. Make sure that there's only one json object in the output, and it should be the reformulated state. Maintain the same structure as your previous formulation. The reply should be in the format:
<reply>
YOUR REPLY TEXT
</reply>
"""


def extract_comment(comment: str) -> dict:
    """Extract the JSON comment block from a critic response."""
    import re
    text = comment.strip()
    # Prefer the first complete JSON object and support nested braces.
    json_start = text.find("{")
    if json_start == -1:
        raise ValueError("No JSON object found in the comment.")
    cnt, i = 0, json_start
    for j, c in enumerate(text[json_start:]):
        if c == "{":
            cnt += 1
        elif c == "}":
            cnt -= 1
            if cnt == 0:
                i = json_start + j + 1
                break
    json_str = text[json_start:i]
    # Remove inline comments.
    json_str = re.sub(r"//.*", "", json_str)
    # Remove trailing commas.
    json_str = re.sub(r",\s*([}\]])", r"\1", json_str)
    return json.loads(json_str)


def extract_reformulation(reformulation: str) -> dict:
    """Extract the reformulated JSON state from a reformulation response."""
    return extract_json_from_end(reformulation)


def extract_reply(reformulation: str) -> str:
    """Extract the reply section from a reformulation response."""
    match = re.search(r"<reply>(.*?)</reply>", reformulation, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        print("[WARNING]: No reply found in the reformulation.")
        return ""
