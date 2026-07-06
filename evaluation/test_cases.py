# evaluation/test_cases.py
# ============================================================
# GROUND TRUTH DATASET
# ============================================================
# Hand-crafted test cases targeting specific RAG weaknesses.
# Each test case defines the EXACT doc_ids that must be
# retrieved to perfectly answer the query.

TEST_CASES = [
    {
        "query": "What is the recommended eGFR threshold for metformin use according to the guidelines?",
        "relevant_doc_ids": ["GL-001", "GL-049", "PM-103"],
        "trap_doc_ids": [],
        "scenario": "conflicting_evidence",
        "reference_answer": "Guidelines conflict: GL-001 and PM-103 say it is safe above 30 mL/min, but GL-049 says it is contraindicated below 60 mL/min."
    },
    {
        "query": "What medications was patient P-170 discharged on and when is their follow-up?",
        "relevant_doc_ids": ["DS-170"],
        "trap_doc_ids": [],
        "scenario": "buried_answer",
        "reference_answer": "Patient P-170 was discharged on insulin glargine 20 units nightly, warfarin 5 mg daily, lisinopril 10 mg daily. Follow-up is with pulmonology in 2 weeks and nephrology in 5 weeks."
    },
    {
        "query": "What are the discharge instructions for the 38M patient with type 1 diabetes mellitus?",
        "relevant_doc_ids": ["DS-152"],
        "trap_doc_ids": ["DS-001", "DS-100", "DS-124"], # Other T1DM patients
        "scenario": "similar_disease",
        "reference_answer": "Discharged on warfarin 5 mg daily, albuterol 2 puffs q4h PRN, insulin glargine 20 units nightly. Follow up with pulmonology in 3 weeks and internal medicine in 6 weeks."
    },
    {
        "query": "Advise follow up for 50F with type 2 diabetes mellitus, heart failure with reduced ejection fraction, and CKD stage 3.",
        "relevant_doc_ids": ["DS-176"],
        "trap_doc_ids": ["DS-150", "DS-090", "DS-084", "DS-158"], # Other patients with same exact diseases
        "scenario": "conflicting_evidence",
        "reference_answer": "Follow-up with cardiology in 1 week and pulmonology in 3 weeks."
    },
    {
        "query": "What is the discharge creatinine and eGFR for patient P-001?",
        "relevant_doc_ids": ["DS-001"],
        "trap_doc_ids": [],
        "scenario": "simple_retrieval",
        "reference_answer": "Creatinine at discharge: 2.0 mg/dL. eGFR: 84 mL/min."
    },
    {
        "query": "What is the recommended eGFR threshold for metformin use?",
        "relevant_doc_ids": ["GL-001", "GL-049", "PM-103"],
        "trap_doc_ids": [],
        "scenario": "guideline_synthesis",
        "reference_answer": "The threshold is either above 30 mL/min (GL-001, PM-103) or above 60 mL/min (GL-049)."
    },
    {
        "query": "Is patient P-160 taking methotrexate or metformin?",
        "relevant_doc_ids": ["DS-160"],
        "trap_doc_ids": ["PM-103"],
        "scenario": "similar_medication",
        "reference_answer": "Patient P-160 is taking both metformin 500 mg BID and methotrexate 10 mg weekly."
    },
    {
        "query": "What is the follow-up plan for the 69M with heart failure with preserved ejection fraction (P-166)?",
        "relevant_doc_ids": ["DS-166"],
        "trap_doc_ids": ["DS-022", "DS-046", "DS-072", "DS-089", "DS-096", "DS-135"],
        "scenario": "overlapping_concepts",
        "reference_answer": "Follow-up with pulmonology in 2 weeks and primary care in 4 weeks."
    },
    {
        "query": "What medications is patient P-180 on?",
        "relevant_doc_ids": ["DS-180"],
        "trap_doc_ids": ["DS-176", "DS-150", "DS-090", "DS-084", "DS-158"], 
        "scenario": "conflicting_evidence",
        "reference_answer": "Patient P-180 is on apixaban 5 mg BID, empagliflozin 10 mg daily, and furosemide 40 mg daily."
    },
    {
        "query": "What are the common side effects of lisinopril mentioned in PubMed abstracts?",
        "relevant_doc_ids": [], 
        "trap_doc_ids": ["DS-001", "DS-100", "DS-170"], 
        "scenario": "missing_evidence",
        "reference_answer": "There is no information regarding the side effects of lisinopril in the PubMed abstracts provided."
    },
    {
        "query": "What is the eGFR of the 72M patient with CKD stage 4 (P-170)?",
        "relevant_doc_ids": ["DS-170"],
        "trap_doc_ids": [],
        "scenario": "buried_answer",
        "reference_answer": "The eGFR is 30 mL/min."
    },
    {
        "query": "Is empagliflozin prescribed for patient P-001?",
        "relevant_doc_ids": ["DS-001"],
        "trap_doc_ids": ["DS-166", "DS-180"], 
        "scenario": "simple_retrieval",
        "reference_answer": "Yes, empagliflozin 10 mg daily is prescribed for patient P-001."
    },
    {
        "query": "Which patients were advised to follow up with nephrology?",
        "relevant_doc_ids": ["DS-170", "DS-100"],
        "trap_doc_ids": [],
        "scenario": "multi_hop",
        "reference_answer": "Patient P-170 (in 5 weeks) and patient P-100 (in 4 weeks) were advised to follow up with nephrology."
    },
    {
        "query": "According to guidelines, should metformin be given if eGFR is 45?",
        "relevant_doc_ids": ["GL-001", "GL-049"],
        "trap_doc_ids": [],
        "scenario": "guideline_synthesis",
        "reference_answer": "There is conflicting guidance. GL-001 suggests it may be continued with monitoring, while GL-049 states it is contraindicated below 60."
    },
    {
        "query": "What is patient P-152's creatinine level?",
        "relevant_doc_ids": ["DS-152"],
        "trap_doc_ids": ["DS-001", "DS-100", "DS-111"], 
        "scenario": "simple_retrieval",
        "reference_answer": "Creatinine at discharge is 0.9 mg/dL."
    }
]
