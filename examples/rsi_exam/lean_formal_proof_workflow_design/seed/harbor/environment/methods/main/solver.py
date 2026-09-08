"""Source-derived seed: propose, compile, repair once, and recompile."""
from __future__ import annotations

from prompt_utils import extract_proof, proposal_messages, repair_messages, safe_call


def answer_batch(examples, llm, lean, budget):
    proofs = []
    for example in examples:
        response = safe_call(llm, proposal_messages(example), max_tokens=6500)
        proof = extract_proof(response)
        checked = lean(proof)
        if not checked["ok"]:
            response = safe_call(
                llm,
                repair_messages(example, proof, checked.get("diagnostics", "")),
                max_tokens=5500,
            )
            proof = extract_proof(response)
            lean(proof)
        proofs.append(proof)
    return proofs
