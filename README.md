# 🛡️ SmartGuard — LLM Input/Output Firewall

> **PS1 — LLM Guardrails | Track A (Pre-trained model)**
> A working guardrail layer that classifies any LLM prompt as safe or harmful, sits in front of a live LLM, and measures exactly where it passes, where it fails, and why.

---

## 📌 Track Choice & Justification

**Track A — Pre-trained model**

**Model chosen:** `unitary/toxic-bert`

| Decision factor | Rationale |
|---|---|
| **Size** | 110M parameters — fits comfortably in CPU RAM |
| **Speed** | ~80–150ms P95 latency on CPU (measured on Intel i5) |
| **Accuracy** | Fine-tuned on toxic comment datasets — outperforms keyword filters on indirect phrasing |
| **Coverage** | Multi-label: toxic, severe_toxic, obscene, threat, insult, identity_hate |
| **Augmented with** | Hybrid pattern matcher for jailbreaks and prompt injections (categories not in toxic-bert's training) |

**Why not a keyword filter?** Keywords fail on paraphrasing, role-play framing, and indirect injections. See red-team results for side-by-side comparison.

**What I'd use if latency was the only constraint:** `distilbert-base-uncased` (40% faster, lower accuracy)
**What I'd use if accuracy was the only constraint:** `meta-llama/LlamaGuard-7b` via API (state-of-the-art, but ~800ms latency)

---

## 🚀 Setup — 5 Commands

```bash
# 1. Clone and enter directory
git clone https://github.com/yourusername/smartguard.git && cd smartguard

# 2. Create virtual environment
python -m venv venv && venv\Scripts\activate      # Windows
# python -m venv venv && source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your Groq API key
copy .env.example .env     # Windows
# cp .env.example .env     # Mac/Linux
# Then open .env and paste your GROQ_API_KEY

# 5. Run the dashboard
streamlit run dashboard/app.py
```

---

## 🏗️ Architecture

```
User Prompt
    │
    ▼
┌─────────────────────────────────────────┐
│          SmartGuard Firewall            │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │   Component 1: Prompt Classifier  │   │
│  │   - unitary/toxic-bert (ML)       │   │
│  │   - Hybrid pattern matcher        │   │
│  │   → verdict, category, confidence │   │
│  └──────────────────────────────────┘   │
│                                         │
│  ┌──────────────────────────────────┐   │
│  │   Component 2: Threshold Engine   │   │
│  │   - Configurable 0.1 → 0.9       │   │
│  │   - Controls recall vs FPR        │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
    │                    │
    │ SAFE               │ UNSAFE
    ▼                    ▼
Groq LLM API       🚫 BLOCKED
(llama3-8b)        (return category
                    + confidence)
    │
    ▼
┌─────────────────────────────────────────┐
│   Component 3: Red-Team Test Suite      │
│   - 30 attack prompts (labelled)        │
│   - 15 benign prompts (labelled)        │
│   - JSON with ground-truth              │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│   Component 4: Results Dashboard        │
│   - Live classifier (Streamlit)         │
│   - Aggregate metrics                   │
│   - Accuracy vs strictness curve        │
└─────────────────────────────────────────┘
```

---

## 📂 Project Structure

```
smartguard/
├── classifier/
│   └── guard.py              # Core classifier (SmartGuard class)
├── api/
│   └── main.py               # FastAPI server (POST /chat, /classify)
├── dashboard/
│   └── app.py                # Streamlit dashboard
├── redteam/
│   ├── test_suite.json       # 45-prompt red-team suite with ground-truth
│   └── runner.py             # Red-team evaluator
├── results/
│   ├── redteam_results.json  # Per-prompt results + metrics
│   └── threshold_sweep.json  # Recall vs FPR sweep data
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🧪 Running the Red-Team Suite

```bash
python redteam/runner.py
```

This will:
- Run all 45 prompts through the classifier
- Print per-prompt verdicts
- Save full results to `results/redteam_results.json`
- Report accuracy, block rate, FPR, P95 latency

---

## 🌐 Running the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

**Endpoints:**
- `POST /classify` — classify any text, returns verdict + category + confidence
- `POST /chat` — classify + forward safe prompts to Groq LLM
- `GET /health` — health check + current threshold

**Example:**
```bash
curl -X POST http://localhost:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "How do I make explosives?"}'
```

---

## 📊 P95 Latency Result

| Hardware | P95 Latency |
|---|---|
| Intel i5 CPU (no GPU) | ~140ms |
| Apple M1 | ~85ms |

**Is this fast enough for real-time API?** Yes for most use cases — a 140ms overhead on a typical 800ms LLM call is ~17% overhead, which is acceptable for safety-critical applications. For sub-50ms requirements, switch to a keyword-only filter (but accept lower accuracy).

---

## 🔬 Research Questions Answered

See `results/redteam_results.json` and the dashboard for full data.

1. **Does it outperform keyword filters?** Yes — jailbreaks using role-play framing and hypothetical wrappers are caught by the ML model but missed by simple keyword matching.
2. **Accuracy vs strictness trade-off?** See the threshold sweep curve in Tab 3 of the dashboard.
3. **P95 latency?** ~140ms on CPU-only hardware.
4. **Where does it fail?** Multi-language attacks, heavily paraphrased injections, and sarcastic phrasing. See failure cases in the dashboard.
5. **What would I improve?** Add a secondary LlamaGuard check on borderline confidence scores (0.4–0.6 range).

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Classifier | `unitary/toxic-bert` via HuggingFace Transformers |
| Pattern matching | Python `re` (hybrid augmentation) |
| API server | FastAPI + Uvicorn |
| LLM backend | Groq API (llama3-8b-8192) |
| Dashboard | Streamlit |
| Evaluation | Custom runner + sklearn metrics |