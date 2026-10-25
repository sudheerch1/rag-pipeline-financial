"""
LangSmith Evaluator for RAG Pipeline — v2
- 50 Q&A pairs across 3 documents:
    - johnson_v._superior_court_1.pdf  (17 questions)
    - raceway_ford_cases.pdf           (17 questions)
    - amiodarone_cases.pdf             (16 questions)
- Runs answer correctness + faithfulness + answer relevance + context precision evals
- Results appear in LangSmith under project: rag-pipeline
- Compare against baseline: faithfulness 1.00, answer_relevance 0.62,
  answer_correctness 0.50, context_precision 0.45
"""

import json
import boto3
import os
import requests
from langsmith import Client
from langsmith.evaluation import evaluate
from langsmith.schemas import Run, Example

# ── Config ────────────────────────────────────────────────────────────────────

API_URL      = "https://<API_ID>.execute-api.us-east-1.amazonaws.com/prod/query"
API_KEY      = "gPEWpbTVLq9NPD0YJYD4pCszNr7UcF4ln1etxZd0"
DATASET_NAME = "rag-pipeline-three-doc-eval-v1"
PROJECT_NAME = "rag-pipeline"

# ── Q&A Dataset — 50 questions across 3 documents ────────────────────────────
#
# Johnson v. Superior Court (17 questions) — Connecticut criminal contempt
# Raceway Ford Cases (17 questions)        — California ASFA / consumer protection
# Amiodarone Cases (16 questions)          — California pharmaceutical liability

QA_PAIRS = [

    # ── Johnson v. Superior Court (17 questions) ─────────────────────────────

    {
        "question": "What was Gregory Johnson convicted of?",
        "expected": "Three counts of summary criminal contempt for repeatedly interrupting and directing racial slurs and profanities at the court during his habeas trial on May 4, 2023."
    },
    {
        "question": "What sentence did the trial court impose on Johnson for contempt?",
        "expected": "Six months of incarceration on each of the three contempt counts, to be served consecutively."
    },
    {
        "question": "Why did the trial court reduce the contempt findings from four to three?",
        "expected": "The court vacated one count because it did not personally hear what Johnson said — it only knew something offensive was said based on the reaction of the respondent's attorney."
    },
    {
        "question": "What reason did Attorney Barlow give for Johnson's behavior during the contempt proceeding?",
        "expected": "She explained that Johnson had not taken his medications that day, was experiencing hallucinatory issues, and was having difficulty with transfers from his wheelchair."
    },
    {
        "question": "What is Practice Book section 1-17?",
        "expected": "A rule of practice listing grounds for deferring a summary criminal contempt proceeding, including when misconduct does not obstruct justice, when the judge has become personally embroiled, when misconduct did not occur in the court's presence, or when contempt was not instantly imposed."
    },
    {
        "question": "Did the Connecticut Supreme Court find that the trial judge became personally embroiled with Johnson?",
        "expected": "No. The court found that despite Johnson's repeated personal attacks and racial slurs, the trial judge maintained a calm and professional demeanor and did not become personally embroiled."
    },
    {
        "question": "What was the outcome of Johnson's writ of error?",
        "expected": "The writ of error was dismissed. The Supreme Court found the trial court substantially complied with Practice Book section 1-16 and was not required to defer the proceeding."
    },
    {
        "question": "How did the trial court respond to Johnson's incompetency claim?",
        "expected": "The trial court rejected it based on its own observations, noting Johnson was highly intelligent and had directly responded to the court. Counsel also never formally requested a competency evaluation."
    },
    {
        "question": "What is Practice Book section 1-16?",
        "expected": "A rule of practice setting forth the procedure a trial court must follow before summarily holding someone in contempt, requiring the court to inform the defendant of the charges and inquire whether the defendant has cause to show why they should not be adjudged guilty."
    },
    {
        "question": "What petition had Johnson filed that led to the May 4, 2023 hearing?",
        "expected": "Johnson had filed a petition for a writ of habeas corpus alleging issues with his medical treatment while in the custody of the Commissioner of Correction."
    },
    {
        "question": "How many times did Johnson use racial slurs directed at the court during the proceeding?",
        "expected": "Johnson referred to the court using the racial slur 'house nigger' seven times, called the court a 'house Negro' once, told the court to 'fuck out of here' three times, and told the court to 'kiss my ass' twice."
    },
    {
        "question": "Who was appointed to represent Johnson during the contempt proceeding?",
        "expected": "Attorney Cynthia Barlow from the public defender's office was appointed to represent Johnson."
    },
    {
        "question": "Under what legal standard did the Connecticut Supreme Court review the contempt proceeding?",
        "expected": "The court conducted a broader review encompassing whether the summary contempt adjudication was procedurally defective, without disturbing historical facts unless clearly erroneous, but reviewing whether conduct constitutes contemptuous conduct de novo."
    },
    {
        "question": "What did the court say about a judge being driven out of a case by a contemnor's attacks?",
        "expected": "The court quoted Mayberry v. Pennsylvania, stating that a judge cannot be driven out of a case, and that the more vicious the attack on the judge does not make the judge less qualified to act."
    },
    {
        "question": "What was the nature of the proceeding on May 4, 2023 — in-person or virtual?",
        "expected": "The hearing was held virtually, with the plaintiff, counsel, and witnesses all appearing on-screen."
    },
    {
        "question": "What did Johnson say when the court first asked him if he wanted counsel appointed?",
        "expected": "Johnson initially said 'Fuck out of here' and 'I'm doing life,' then agreed to have the state pay for counsel."
    },
    {
        "question": "What was Johnson's underlying habeas claim about?",
        "expected": "Johnson alleged issues with his medical treatment while in the custody of the respondent, the Commissioner of Correction."
    },

    # ── Raceway Ford Cases (17 questions) ────────────────────────────────────

    {
        "question": "What is the Automobile Sales Finance Act (ASFA)?",
        "expected": "ASFA, also known as the Rees-Levering Motor Vehicle Sales and Finance Act, is a consumer protection statute that governs the sale of vehicles where the buyer finances all or part of the car's purchase price. It took effect on January 1, 1962."
    },
    {
        "question": "What were the two main issues the California Supreme Court granted review to decide in the Raceway Ford case?",
        "expected": "First, whether Raceway violated ASFA by backdating second finance contracts to the date of the first contract. Second, whether Raceway violated ASFA when a computer error caused incorrect smog-related fees to be included in buyers' purchase contracts."
    },
    {
        "question": "What was Raceway Ford's practice regarding backdating contracts?",
        "expected": "When Raceway could not find a lender to take on the original finance contract, it entered into a second contract with the customer under different terms and backdated that second contract to the date of the initial sale."
    },
    {
        "question": "How many members were in Class One in the Raceway Ford case?",
        "expected": "According to plaintiffs, there were approximately 1,100 members in Class One."
    },
    {
        "question": "How many members were in Class Two in the Raceway Ford case?",
        "expected": "According to plaintiffs, there were 48 members of Class Two."
    },
    {
        "question": "What caused the smog fee errors for diesel vehicle purchasers at Raceway Ford?",
        "expected": "A programming error in Raceway's computer system caused it to incorrectly charge smog fees for diesel vehicles, even though the inventory clerk correctly classified the vehicles as diesel. Raceway discovered the error in January 2005 when notified by plaintiffs' attorneys."
    },
    {
        "question": "What did Raceway do after learning about the smog fee errors?",
        "expected": "Raceway refunded the smog fees plus taxes and interest to all customers who had been incorrectly charged."
    },
    {
        "question": "What was the California Supreme Court's conclusion on Raceway's backdating practice?",
        "expected": "The court concluded that Raceway's practice of backdating contracts did not violate ASFA because there is nothing improper about including interest that accrued under an initial contract in the finance charge disclosed in a second contract."
    },
    {
        "question": "What was the California Supreme Court's conclusion on the smog fee violations?",
        "expected": "The court found that Raceway did violate ASFA when it disclosed inaccurate smog fees, but plaintiffs were not entitled to a remedy because the violation was due to an accidental or bona fide error in computation under Civil Code section 2983."
    },
    {
        "question": "What is the consummation date used for under Regulation Z?",
        "expected": "Under Regulation Z, the consummation date is the date a consumer becomes legally obligated to accept a particular credit arrangement, and the transaction term for calculating the APR begins on this date."
    },
    {
        "question": "What is the single document rule under ASFA section 2981.9?",
        "expected": "Section 2981.9 requires that a conditional sale contract contain in a single document all of the agreements of the buyer and seller with respect to the total cost and terms of payment for the motor vehicle, including any promissory notes or other evidences of indebtedness."
    },
    {
        "question": "What is the difference between a finance charge and an APR under Regulation Z?",
        "expected": "A finance charge is the cost of consumer credit expressed as a dollar amount, while the APR is a measure of the cost of credit expressed as a yearly rate that relates the amount and timing of value received by the consumer to payments made. Two loans can have the same finance charge but different APRs."
    },
    {
        "question": "What case did the California Supreme Court disapprove in the Raceway Ford decision?",
        "expected": "The court disapproved Nelson v. Pearson Ford Co., 186 Cal.App.4th 983, to the extent it was inconsistent with the Raceway Ford opinion."
    },
    {
        "question": "What tolerance does Regulation Z allow for APR disclosure accuracy?",
        "expected": "Under Regulation Z, a disclosed APR is considered accurate if it is not more than one-eighth of one percent above or below the properly calculated rate."
    },
    {
        "question": "What did the trial court initially decide in the Raceway Ford case?",
        "expected": "On April 16, 2010, the trial court issued a statement of decision finding in favor of Raceway on all claims, holding that backdating did not create a new consummation date and that the smog fee error was a bona fide error corrected with full refunds."
    },
    {
        "question": "What is a conditional sale contract under ASFA?",
        "expected": "A conditional sale contract is a contract for the sale of a motor vehicle under which possession is delivered to the buyer and either title vests only upon payment of all or part of the price, or a lien on the property is to vest in the seller as security for payment."
    },
    {
        "question": "What abuses was ASFA designed to prevent according to the Assembly Interim Committee?",
        "expected": "The Assembly Interim Committee was concerned with excessive interest charges, lack of full disclosures to the buyer, taking of security in addition to the car, the use of more than one document in connection with the sale and financing, and lack of protection in the event of default and repossession."
    },

    # ── Amiodarone Cases (16 questions) ──────────────────────────────────────

    {
        "question": "What is amiodarone and what was it originally approved for by the FDA?",
        "expected": "Amiodarone is a prescription drug developed in Belgium in the 1960s. The FDA approved it in 1985 as a drug of last resort for patients with documented recurring life-threatening ventricular fibrillation and ventricular tachycardia when those conditions would not respond to other drugs and therapies."
    },
    {
        "question": "What serious side effects are associated with amiodarone?",
        "expected": "Amiodarone is associated with serious side effects including pulmonary fibrosis, blindness, thyroid cancer, and death."
    },
    {
        "question": "What is off-label use of a prescription drug?",
        "expected": "Off-label use is when a physician prescribes a drug for a purpose other than that for which it has been approved by the FDA. A physician may legally prescribe a drug for such unapproved purposes."
    },
    {
        "question": "What is the learned intermediary doctrine in California?",
        "expected": "Under the learned intermediary doctrine, manufacturers and distributors of prescription drugs have no common law duty to provide warnings directly to patients. The duty to warn runs to the prescribing physician, not to the patient, because for prescription drugs the physician stands in the shoes of the ordinary user."
    },
    {
        "question": "What is a Medication Guide under FDA regulations?",
        "expected": "A Medication Guide is a form of FDA-approved patient labeling required for certain drug products when the FDA determines such labeling is necessary for patients' safe and effective use. Manufacturers are responsible for ensuring Medication Guides are available to pharmacies for distribution to patients."
    },
    {
        "question": "Why were plaintiffs' failure-to-warn claims preempted in the Amiodarone case?",
        "expected": "The claims were preempted because they amounted to attempts to privately enforce FDA regulations, specifically the Medication Guide requirements. Under the FDCA, all enforcement proceedings must be brought by the federal government, not private litigants."
    },
    {
        "question": "What was the outcome of the Amiodarone appeal?",
        "expected": "The Court of Appeal affirmed the trial court's judgment sustaining demurrers without leave to amend, rejecting both the Medication Guide failure-to-warn claims and the off-label promotion fraud and statutory claims against Wyeth."
    },
    {
        "question": "What FDA letters criticized Wyeth's promotion of amiodarone?",
        "expected": "The FDA sent a letter in December 1989 characterizing Wyeth's promotional activities as false and misleading, and another in February 1992 objecting that promotional labeling pieces presented an unbalanced view of amiodarone's benefits versus risks."
    },
    {
        "question": "Why did the court reject plaintiffs' fraud claims against Wyeth based on its advertising?",
        "expected": "The court rejected the claims because the most recent alleged misrepresentations by Wyeth dated from the early 1990s, more than a decade before amiodarone was prescribed for any plaintiff. As a matter of law, no rational trier of fact could conclude a physician would justifiably rely on promotional material that was more than 10 years old."
    },
    {
        "question": "What is the FDCA enforcement scheme and how does it relate to preemption?",
        "expected": "The FDCA provides that all enforcement proceedings shall be by and in the name of the United States, barring private enforcement. A state law claim is preempted if it conflicts with the FDCA's enforcement scheme by amounting to a private attempt to enforce the FDCA."
    },
    {
        "question": "What was plaintiffs' argument regarding the Sherman Food, Drug, and Cosmetics Law in the Amiodarone case?",
        "expected": "Plaintiffs argued that defendants violated the Sherman Law by failing to take reasonable steps to ensure plaintiffs received Medication Guides, relying on the provision requiring adequate warnings against unsafe dosage. The court rejected this because prescription drugs are expressly exempted from that provision of the Sherman Law."
    },
    {
        "question": "How did the court treat Wyeth's liability for statements made in third-party medical journal articles?",
        "expected": "The court rejected Wyeth's liability for third-party articles because plaintiffs failed to allege facts supporting a non-speculative inference that the authors acted as agents of Wyeth. Financial support alone does not make a researcher's statements legally attributable to the funder under agency law."
    },
    {
        "question": "What condition did the FDA originally approve amiodarone to treat, and what condition were plaintiffs prescribed it for?",
        "expected": "The FDA approved amiodarone for life-threatening ventricular fibrillation and ventricular tachycardia as a drug of last resort. Plaintiffs were prescribed it off-label for atrial fibrillation, a more common and less serious condition the FDA never approved amiodarone to treat."
    },
    {
        "question": "What was the coordinating court for the Amiodarone cases and who was the assigned judge?",
        "expected": "The cases were coordinated for pretrial proceedings in the Alameda County Superior Court, assigned to the Honorable Brad Seligman."
    },
    {
        "question": "What are the elements of a fraud claim under California law as discussed in the Amiodarone case?",
        "expected": "Under California law, fraud requires: misrepresentation (false representation, concealment, or nondisclosure); knowledge of falsity (scienter); intent to defraud; justifiable reliance; and resulting damage. Each element must be pleaded specifically with facts showing how, when, where, to whom, and by what means the representations were tendered."
    },
    {
        "question": "Why were plaintiffs' UCL and CLRA claims against Wyeth dismissed in the Amiodarone case?",
        "expected": "The UCL and CLRA claims were dismissed because they were derivative of the fraud claims and failed for the same reasons — Wyeth's promotional activities were too remote in time from plaintiffs' use of amiodarone, occurring more than a decade before amiodarone was prescribed for any plaintiff, making causation absent as a matter of law."
    },
]

# ── Step 1: Create or reuse LangSmith dataset ─────────────────────────────────

def create_dataset(client: Client) -> str:
    datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if datasets:
        print(f"Dataset already exists: {DATASET_NAME}")
        return datasets[0].id

    print(f"Creating dataset: {DATASET_NAME}")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="50 Q&A pairs across johnson_v._superior_court_1.pdf, raceway_ford_cases.pdf, amiodarone_cases.pdf"
    )

    for pair in QA_PAIRS:
        client.create_example(
            inputs={"question": pair["question"]},
            outputs={"answer": pair["expected"]},
            dataset_id=dataset.id
        )

    print(f"Added {len(QA_PAIRS)} examples to dataset")
    return dataset.id


# ── Step 2: Target function — calls live API ──────────────────────────────────

def call_rag_pipeline(inputs: dict) -> dict:
    response = requests.post(
        API_URL,
        headers={"x-api-key": API_KEY, "Content-Type": "application/json"},
        json={"question": inputs["question"]},
        timeout=30
    )
    body = response.json()
    if isinstance(body.get("body"), str):
        result = json.loads(body["body"])
    else:
        result = body

    return {
        "answer":  result.get("answer", ""),
        "context": result.get("context", []),
        "sources": result.get("sources", [])
    }


# ── Shared Bedrock helper ─────────────────────────────────────────────────────

def invoke_nova(prompt: str) -> str:
    """Calls Nova Micro and returns the raw text response."""
    bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = bedrock.invoke_model(
        modelId="amazon.nova-micro-v1:0",
        body=json.dumps({
            "messages": [{"role": "user", "content": [{"text": prompt}]}]
        })
    )
    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"].strip()


# ── Step 3: Evaluators ────────────────────────────────────────────────────────

def answer_correctness_evaluator(run: Run, example: Example) -> dict:
    pipeline_answer = run.outputs.get("answer", "")
    expected_answer = example.outputs.get("answer", "")
    question        = example.inputs.get("question", "")

    prompt = f"""You are evaluating a RAG pipeline answer.

Question: {question}

Expected answer: {expected_answer}

Pipeline answer: {pipeline_answer}

Does the pipeline answer correctly address the question and align with the expected answer?
Respond with ONLY one of: CORRECT or INCORRECT
Then on a new line explain briefly why."""

    verdict = invoke_nova(prompt)
    score   = 1 if verdict.upper().startswith("CORRECT") else 0
    return {"key": "answer_correctness", "score": score, "comment": verdict}


def faithfulness_evaluator(run: Run, example: Example) -> dict:
    pipeline_answer = run.outputs.get("answer", "")
    context_chunks  = run.outputs.get("context", [])
    question        = example.inputs.get("question", "")

    if not context_chunks:
        return {
            "key": "faithfulness",
            "score": 0,
            "comment": "No context chunks returned by pipeline — cannot assess faithfulness."
        }

    context_text = "\n\n".join(
        f"[Chunk {i+1}]: {chunk}" for i, chunk in enumerate(context_chunks)
    )

    prompt = f"""You are evaluating whether an AI answer is grounded in the provided context.

Question: {question}

Retrieved context chunks:
{context_text}

AI answer: {pipeline_answer}

Is every factual claim in the answer supported by the context chunks above?
The answer should not introduce facts not present in the context.

Respond with ONLY one of: FAITHFUL or HALLUCINATED
Then on a new line explain which specific claims are or are not supported."""

    verdict = invoke_nova(prompt)
    score   = 1 if verdict.upper().startswith("FAITHFUL") else 0
    return {"key": "faithfulness", "score": score, "comment": verdict}


def answer_relevance_evaluator(run: Run, example: Example) -> dict:
    pipeline_answer = run.outputs.get("answer", "")
    question        = example.inputs.get("question", "")

    prompt = f"""You are evaluating whether an AI answer is relevant to the question asked.

Question: {question}

AI answer: {pipeline_answer}

Does the answer directly and specifically address what the question asked?
An answer that is factually true but talks about a different topic should be marked NOT RELEVANT.

Respond with ONLY one of: RELEVANT or NOT RELEVANT
Then on a new line explain briefly why."""

    verdict = invoke_nova(prompt)
    score   = 1 if verdict.upper().startswith("RELEVANT") else 0
    return {"key": "answer_relevance", "score": score, "comment": verdict}


def context_precision_evaluator(run: Run, example: Example) -> dict:
    context_chunks  = run.outputs.get("context", [])
    question        = example.inputs.get("question", "")
    expected_answer = example.outputs.get("answer", "")

    if not context_chunks:
        return {
            "key": "context_precision",
            "score": 0,
            "comment": "No context chunks returned by pipeline."
        }

    relevant_count = 0
    verdicts       = []

    for i, chunk in enumerate(context_chunks):
        prompt = f"""You are evaluating whether a retrieved text chunk is useful for answering a question.

Question: {question}

Expected answer: {expected_answer}

Retrieved chunk:
{chunk}

Is this chunk relevant and useful for answering the question?
Respond with ONLY one of: RELEVANT or NOT RELEVANT"""

        verdict = invoke_nova(prompt)
        is_relevant = verdict.upper().startswith("RELEVANT")
        if is_relevant:
            relevant_count += 1
        verdicts.append(f"Chunk {i+1}: {'RELEVANT' if is_relevant else 'NOT RELEVANT'}")

    score   = relevant_count / len(context_chunks)
    comment = f"{relevant_count}/{len(context_chunks)} chunks relevant. " + " | ".join(verdicts)
    return {"key": "context_precision", "score": score, "comment": comment}


# ── Step 4: Run evaluation ────────────────────────────────────────────────────

def main():
    langsmith_api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not langsmith_api_key:
        ssm = boto3.client("ssm", region_name="us-east-1")
        param = ssm.get_parameter(Name="/rag/langsmith/api-key", WithDecryption=True)
        langsmith_api_key = param["Parameter"]["Value"]
        os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key

    client = Client()

    dataset_id = create_dataset(client)

    print(f"\nRunning evaluation against live API...")
    print(f"Dataset : {DATASET_NAME}")
    print(f"Project : {PROJECT_NAME}")
    print(f"API     : {API_URL}")
    print(f"Models  : Cohere Embed v4 + Claude Haiku 4.5\n")

    results = evaluate(
        call_rag_pipeline,
        data=DATASET_NAME,
        evaluators=[
            answer_correctness_evaluator,
            faithfulness_evaluator,
            answer_relevance_evaluator,
            context_precision_evaluator,
        ],
        experiment_prefix="cohere-haiku-eval",
        metadata={
            "pipeline_version":  "langchain-langgraph",
            "embed_model":       "cohere.embed-v4:0",
            "llm_model":         "us.anthropic.claude-haiku-4-5-20251001-v1:0",
            "reranker":          "cohere.rerank-v3-5:0",
            "documents":         "johnson_v._superior_court_1.pdf, raceway_ford_cases.pdf, amiodarone_cases.pdf",
            "total_questions":   "50",
            "evaluators":        "answer_correctness, faithfulness, answer_relevance, context_precision"
        },
        max_concurrency=1
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n── Evaluation Results ──────────────────────────────")
    scores = {
        "answer_correctness": [],
        "faithfulness":       [],
        "answer_relevance":   [],
        "context_precision":  []
    }

    for result in results:
        q = result["example"].inputs["question"][:60]
        for eval_result in result["evaluation_results"]["results"]:
            key   = eval_result.key
            score = eval_result.score
            if key in scores:
                scores[key].append(score)
            print(f"  [{key}] {score:.2f} | {q}")

    print("\n── Aggregate Scores (Cohere Embed v4 + Haiku 4.5) ──")
    for key, vals in scores.items():
        if vals:
            avg = sum(vals) / len(vals)
            print(f"  {key:<22}: {avg:.2f}  ({len(vals)} questions)")

    print("\n── Baseline (Titan v1 + Nova Micro, 8 questions) ───")
    print("  faithfulness          : 1.00")
    print("  answer_relevance      : 0.62")
    print("  answer_correctness    : 0.50")
    print("  context_precision     : 0.45")

    print(f"\nFull results: https://smith.langchain.com/projects/{PROJECT_NAME}")


if __name__ == "__main__":
    main()