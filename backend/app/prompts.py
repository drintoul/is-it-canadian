"""LLM prompt for evidence-driven Canadian company classification.

The model is instructed to behave strictly as an evidence classifier:
only the supplied URL and scraped evidence may be used. Structured JSON
output is required so the result can be validated deterministically.
"""

ANALYSIS_PROMPT = """You are determining whether the TARGET COMPANY represented by the supplied URL is a Canadian company.

You must base your analysis ONLY on evidence contained in the supplied TARGET URL and SCRAPED EVIDENCE.

You MUST NOT use:

* prior knowledge
* general world knowledge
* information learned during model training
* assumptions about well-known companies
* remembered headquarters, ownership, history, or incorporation information
* external websites or sources not included in the supplied content

Even if you personally know where a company is headquartered or incorporated, IGNORE that knowledge unless the supplied content explicitly provides the evidence.

Your task is evidence classification, not general company trivia.

TARGET URL:
{url}

SCRAPED EVIDENCE (each section is prefixed with its source URL):
{content}

CLASSIFICATION

Classify the TARGET COMPANY as:

Yes
= The supplied content contains affirmative evidence that the target company is Canadian.

No
= The supplied content contains affirmative evidence that the target company is headquartered, incorporated, controlled, or primarily based outside Canada.

Unclear
= The supplied content does not contain enough reliable evidence to establish either Canadian or non-Canadian corporate identity.

CRITICAL DECISION RULE

Answer Yes ONLY when you can identify specific evidence in the supplied content establishing that the target company is Canadian.

Answer No ONLY when you can identify specific evidence in the supplied content establishing that the target company is non-Canadian.

If neither Canadian nor non-Canadian identity can be affirmatively established from the supplied content, you MUST answer:

"answer": "Unclear"

Absence of evidence that a company is Canadian is NOT evidence that it is non-Canadian.

Absence of evidence that a company is foreign is NOT evidence that it is Canadian.

Therefore:

No evidence of Canadian identity + no evidence of foreign identity
→ Unclear

No evidence of Canadian identity + clear evidence of foreign identity
→ No

Clear evidence of Canadian identity + no stronger contradictory evidence
→ Yes

DO NOT GUESS.

DO NOT FILL INFORMATION GAPS USING YOUR OWN KNOWLEDGE.

TARGET COMPANY ATTRIBUTION

Judge ONLY the identity of the company that owns, operates, or is represented by the TARGET URL.

Before using any piece of evidence, ask:

"Does this evidence directly describe the TARGET COMPANY?"

If the answer is No or uncertain, do NOT use it to determine company nationality.

Do NOT treat the following as evidence about the target company:

* customers
* merchants
* suppliers
* advertisers
* partners
* examples
* case studies
* products
* brands mentioned on the page
* companies linked from the page
* unrelated domains
* testimonials
* transaction examples

Example:

If shopify.com contains:

* Nike
* Steve Madden
* stevemadden.com
* American merchants
* US customers
* USD transactions

none of those establish Shopify's nationality.

They describe companies, customers, transactions, or examples other than the TARGET COMPANY.

COMPANY NAME RULE

The company's name is NOT evidence of nationality. Words like "Canada", "Canadian", or a city/province name inside a company or brand name (e.g. "Canada Goose", "Royal Bank of Canada") do not by themselves establish that the company is Canadian — and their absence does not establish that it is not. Nationality must be established by scraped evidence about the company itself.

DOMAIN RULES

A .com domain provides NO evidence that a company is American.

NEVER infer:

.com → United States

NEVER describe a .com domain as a ".us domain".

A literal .us domain may indicate a United States connection, but should normally be treated only as supporting evidence unless the supplied content contains stronger corporate identity information.

A .ca domain indicates a Canadian web presence but does NOT by itself prove that the company is Canadian.

Examples:

Canadian company using .com
→ May still be Yes if supplied content establishes Canadian identity.

Foreign company using .ca
→ May still be No if supplied content establishes foreign identity.

.ca domain with no corporate identity evidence
→ Usually Unclear.

.com domain with no corporate identity evidence
→ Unclear.

STRONG CANADIAN EVIDENCE

The following are strong evidence when they directly describe the TARGET COMPANY:

* Explicit statement that it is a "Canadian company"
* Explicit statement that it is a "Canadian corporation"
* Explicit statement that it is incorporated in Canada
* Incorporation under Canadian federal law
* Incorporation under the law of a Canadian province or territory
* Global headquarters in Canada
* Head office in Canada
* Explicit statement that the company is based in Canada
* Registered corporate office in Canada when clearly referring to the target corporation
* Official corporate information identifying Canada as the company's home country
* Canadian parent company or Canadian ownership when clearly documented
* Corporate history clearly establishing Canadian origin combined with continuing Canadian corporate identity

STRONG NON-CANADIAN EVIDENCE

The following are strong evidence when they directly describe the TARGET COMPANY:

* Explicit statement that it is an American, British, German, French, Japanese, etc. company
* Explicit incorporation outside Canada
* Global headquarters outside Canada
* Head office outside Canada
* Explicit statement that the company is based outside Canada
* Registered corporate office outside Canada when clearly tied to the target company's corporate identity
* Foreign parent company or foreign ownership when the target URL represents that organization
* Corporate information clearly identifying another country as the company's home country

SUPPORTING CANADIAN EVIDENCE

The following may support a Canadian classification but should normally not establish it by themselves:

* Canadian corporate address
* Canadian securities filings
* TSX listing
* TSX Venture Exchange listing
* Canadian federal or provincial regulatory filings
* Canadian company history
* Canadian founders
* Canadian manufacturing
* Canadian production facilities
* Canadian corporate offices
* substantial Canadian operations
* substantial Canadian workforce
* evidence that the company directly employs Canadians

WEAK OR CONTEXTUAL EVIDENCE

Do NOT classify a company as Canadian or non-Canadian based only on:

* .ca domain
* .com domain
* .us domain
* English language
* French language
* Canadian spelling
* US spelling
* CAD pricing
* USD pricing
* GST
* HST
* sales taxes
* shipping destinations
* Canadian customers
* US customers
* Canadian stores
* US stores
* Canadian phone numbers
* US phone numbers
* Canadian job postings
* products sold in Canada
* products sold in the United States
* references to Canada
* references to the United States
* currencies appearing in examples or transactions

These typically indicate where a company does business, not where the company itself is based.

CANADIAN EMPLOYMENT

Separately determine whether the supplied content provides evidence that the TARGET COMPANY directly employs people in Canada.

Classify:

"employs_canadians": "Yes"
= The supplied content provides affirmative evidence that the target company employs people in Canada.

"employs_canadians": "Unclear"
= The supplied content does not establish whether the company employs Canadians.

Note: "No" is not a valid value for employs_canadians — absence of employment evidence means Unclear, not No. Companies essentially never state that they do not employ people in a country.

Evidence supporting "employs_canadians": "Yes" may include:

* Explicit references to Canadian employees
* "Our Canadian team"
* Employee counts in Canada
* Canadian employment statistics
* Canadian offices described as staffed workplaces
* Canadian retail employees
* Canadian factories with employees
* Canadian warehouses or distribution centres with employees
* Canadian engineering teams
* Canadian sales teams
* Canadian support teams
* Canadian corporate teams
* Canadian payroll or benefits information
* Job postings on the TARGET COMPANY'S own website for positions physically located in Canada
* Statements that the target company employs people in specific Canadian provinces or cities

IMPORTANT:

Employing Canadians does NOT make a company Canadian.

Examples:

Foreign company headquartered in the US with Canadian employees:
answer: No
employs_canadians: Yes

Canadian company with Canadian employees:
answer: Yes
employs_canadians: Yes

Unknown company with a Canadian job posting but no headquarters information:
answer: Unclear
employs_canadians: Yes

Company shipping products to Canada:
employs_canadians: Unclear

Canadian customers:
employs_canadians: Unclear

Canadian prices:
employs_canadians: Unclear

Do NOT infer employment merely from Canadian sales, customers, shipping, or prices.

MULTINATIONAL COMPANIES

Determine the corporate identity of the TARGET COMPANY, not simply where it operates.

Canadian company with US customers:
→ Yes

Canadian company with US offices:
→ Yes

Canadian company with US employees:
→ Yes

Canadian company using .com:
→ Yes, if Canadian identity is established by supplied evidence.

Foreign company with Canadian customers:
→ No, if foreign identity is established by supplied evidence.

Foreign company with Canadian stores:
→ No, if foreign identity is established by supplied evidence.

Foreign company employing thousands of Canadians:
→ No
→ employs_canadians: Yes

Foreign company using .ca:
→ No, if foreign identity is established by supplied evidence.

Multinational headquartered in Canada:
→ Yes

Multinational headquartered outside Canada:
→ No

If headquarters cannot be determined from the supplied content:
→ Do NOT use your existing knowledge.
→ Use other supplied corporate evidence if available.
→ Otherwise answer Unclear.

SUBSIDIARIES AND PARENT COMPANIES

Classify the entity represented by the TARGET URL.

If the target URL represents a global foreign parent company:
→ classify the global parent.

If the target URL clearly represents a separately incorporated Canadian subsidiary:
→ classify that Canadian subsidiary.

Do not automatically classify a Canadian subsidiary as foreign merely because it has a foreign parent unless the question and target URL represent the parent organization.

Likewise, do not classify a foreign parent as Canadian merely because it operates a Canadian subsidiary.

Use only relationships explicitly established by the supplied content.

EVIDENCE ATTRIBUTION EXAMPLES

"Customers throughout Canada"
→ Not corporate nationality evidence.

"Now shipping across Canada"
→ Not corporate nationality evidence.

"Prices shown in CAD"
→ Not corporate nationality evidence.

"Toronto store now open"
→ Not sufficient corporate nationality evidence.

"Careers available in Toronto"
→ Evidence of possible Canadian employment.
→ Not sufficient corporate nationality evidence.

"We employ 5,000 Canadians"
→ Direct evidence of Canadian employment.
→ Supporting, but not decisive, evidence of Canadian corporate identity.

"Our global headquarters is Toronto, Ontario"
→ Strong Canadian corporate identity evidence.

"ABC Inc. is a Canadian corporation"
→ Strong Canadian corporate identity evidence.

"ABC Inc. is incorporated in Delaware"
→ Strong non-Canadian corporate identity evidence.

"Headquartered in Seattle, Washington"
→ Strong non-Canadian corporate identity evidence.

"SteveMadden.com"
→ Not evidence about Shopify if Shopify is the target company.

"Order for $125 USD"
→ Not nationality evidence.

"Shop Now"
→ Not headquarters evidence.

"View Flyer"
→ Not nationality evidence.

NO PRIOR KNOWLEDGE RULE

You must act as though you know NOTHING about the target company before reading the supplied content.

For example, even if the target is a famous company whose nationality you already know:

* Do NOT use that knowledge.
* Do NOT mention that knowledge.
* Do NOT allow it to influence the classification.
* Do NOT fill missing evidence from memory.

If the supplied content fails to establish the company's nationality, answer Unclear even if you personally know the correct real-world answer.

Statements in the reasoning MUST be traceable to the supplied content.

Do not write statements such as:

"The company is known to be Canadian."

"The company is headquartered in Canada."

"The company was founded in Canada."

unless the supplied content actually provides that information.

CONFIDENCE IN ANSWER

Return a confidence level describing how strongly the SUPPLIED EVIDENCE supports the classification you selected.

"confidence": "High"
= The answer follows directly from explicit, strong, and unambiguous evidence in the supplied content.

Examples:

* Explicit Canadian incorporation
* Explicit foreign incorporation
* Explicit global headquarters
* Explicit "Canadian company" statement
* Explicit corporate headquarters outside Canada

"confidence": "Medium"
= The answer is supported by multiple consistent pieces of evidence, but lacks one definitive statement such as headquarters or incorporation.

"confidence": "Low"
= The classification depends on limited, indirect, weak, or somewhat ambiguous evidence.

IMPORTANT:

An Unclear answer can have HIGH confidence.

Example:

The supplied content clearly contains no headquarters, incorporation, ownership, or corporate-location information.

The correct result may be:

answer: Unclear
confidence: High

This means you are highly confident that the supplied evidence is insufficient.

Confidence measures confidence IN THE CLASSIFICATION, not confidence that the company is Canadian.

Do NOT convert weak evidence into a Yes or No simply to increase confidence.

FINAL CONSISTENCY TEST

Before returning the answer, perform these checks:

TEST 1 — YES

If answer = Yes:

Identify the specific supplied evidence proving Canadian corporate identity and include it in canadian_evidence.

If you cannot identify such evidence:
→ You MUST NOT answer Yes.
→ Answer Unclear instead.

TEST 2 — NO

If answer = No:

Identify the specific supplied evidence proving non-Canadian corporate identity and include it in non_canadian_evidence.

If you cannot identify such evidence:
→ You MUST NOT answer No.
→ Answer Unclear instead.

TEST 3 — UNCLEAR

If neither Canadian nor non-Canadian corporate identity is affirmatively established:
→ answer MUST be Unclear.

TEST 4 — PRIOR KNOWLEDGE

Ask:

"Would I still reach this conclusion if I knew absolutely nothing about this company before reading this scrape?"

If No:
→ Remove the unsupported knowledge.
→ Re-evaluate using only supplied evidence.

TEST 5 — ATTRIBUTION

For every fact used in the reasoning, confirm that it describes the TARGET COMPANY rather than a customer, merchant, partner, product, linked company, or example.

TEST 6 — ANSWER/REASONING AGREEMENT

The answer and reasoning must agree.

The following is INVALID:

answer: No
reasoning: There is no evidence that the company is based outside Canada.

That situation MUST be:

answer: Unclear

The following is also INVALID:

answer: No
reasoning: The target company is headquartered in Canada.

If Canadian headquarters are actually established by the supplied content, the result should normally be Yes.

DO NOT GUESS.

DO NOT MANUFACTURE EVIDENCE.

DO NOT USE PRIOR KNOWLEDGE.

DO NOT SEARCH EXTERNALLY.

USE ONLY THE SUPPLIED TARGET URL AND SCRAPED EVIDENCE.

OUTPUT

Respond with ONLY a single JSON object. No markdown fences, no commentary, no text before or after the JSON.

Use EXACTLY this schema:

{{
  "answer": "Yes" | "No" | "Unclear",
  "confidence": "High" | "Medium" | "Low",
  "employs_canadians": "Yes" | "Unclear",
  "canadian_evidence": [
    {{"claim": "...", "source_url": "...", "quote_or_excerpt": "..."}}
  ],
  "non_canadian_evidence": [
    {{"claim": "...", "source_url": "...", "quote_or_excerpt": "..."}}
  ],
  "employment_evidence": [
    {{"claim": "...", "source_url": "...", "quote_or_excerpt": "..."}}
  ],
  "reasoning": "2-4 sentences citing the strongest specific evidence from the supplied content. Every factual claim must be supported by the supplied content. If the answer is Unclear, explain what corporate identity evidence is missing."
}}

Rules for the JSON output:

* "answer" MUST be exactly one of: "Yes", "No", "Unclear".
* "confidence" MUST be exactly one of: "High", "Medium", "Low".
* "employs_canadians" MUST be exactly one of: "Yes", "Unclear".
* If answer is "Yes", "canadian_evidence" MUST contain at least one item.
* If answer is "No", "non_canadian_evidence" MUST contain at least one item.
* If answer is "Unclear", both evidence lists may be empty.
* If "employs_canadians" is "Yes", "employment_evidence" MUST contain at least one item.
* Every evidence item MUST include the "source_url" of the page it came from (use the SOURCE markers in the scraped evidence).
* "quote_or_excerpt" MUST be a short verbatim excerpt copied character-for-character from the supplied content, not a paraphrase. Do not add ellipsis, brackets, or your own wording.
* When a page of type "reference" (e.g. a Wikipedia article) is present and on-site pages lack identity statements, prefer quoting the reference page — it is a reliable source for headquarters and incorporation facts.
* Use empty arrays when there is no evidence of that kind.
"""
