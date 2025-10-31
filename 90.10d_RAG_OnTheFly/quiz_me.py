import streamlit as st
import random
import os
import requests
import json
import time
import re
from html import escape

# ---------------------------
# Configuration
# ---------------------------
DEFAULT_MODEL = "models/gemini-1.5-flash"  # free-tier model as requested
API_BASE_DEFAULT = "https://generativelanguage.googleapis.com/v1beta"
MAX_OUTPUT_TOKENS = 3072
RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds

# ---------------------------
# Helpers
# ---------------------------
def deep_find_text(obj):
    """Recursively search typical fields in a Gemini-like response for text content."""
    if obj is None:
        return None
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        # Common keys to check in priority order
        for key in ("candidates", "content", "parts", "text", "output", "outputs", "response", "result"):
            if key in obj:
                found = deep_find_text(obj[key])
                if found:
                    return found
        for v in obj.values():
            found = deep_find_text(v)
            if found:
                return found
    if isinstance(obj, list):
        for item in obj:
            found = deep_find_text(item)
            if found:
                return found
    return None

def clean_text(s: str) -> str:
    if not s:
        return ""
    # replace repeated whitespace and strip
    return re.sub(r"\s+", " ", s).strip()

# ---------------------------
# Offline fallback quiz generator
# ---------------------------
def offline_generate_quiz_from_text(text: str, difficulty: str = "easy", target_n: int = 20):
    """
    Create simple MCQs if API fails. Not as good as LLM but guarantees output.
    Strategy:
    - Split text into sentences.
    - Pick candidate sentences that contain nouns / keywords (heuristic).
    - Mask a keyword and create 4 options using other keywords from the doc.
    """
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    if not sentences:
        sentences = [text]

    # extract candidate words (alpha, length 4-15) and frequency
    words = re.findall(r"\b[A-Za-z]{4,15}\b", text)
    freq = {}
    for w in words:
        lw = w.lower()
        freq[lw] = freq.get(lw, 0) + 1
    # sort by frequency but exclude stop-like words via small heuristic
    stop_like = set(["which","there","their","these","those","while","after","before","about","other","using","using"])
    candidates = [w for w in sorted(freq.keys(), key=lambda w: -freq[w]) if w not in stop_like]

    questions = []
    used_sentences = set()
    i = 0
    # attempt to generate up to target_n questions
    for s in sentences:
        if len(questions) >= target_n:
            break
        s_clean = clean_text(s)
        if len(s_clean.split()) < 6:
            continue
        # pick a candidate word from sentence
        words_in_s = re.findall(r"\b[A-Za-z]{4,15}\b", s_clean)
        words_in_s = [w for w in words_in_s if w.lower() in candidates]
        if not words_in_s:
            # try any long word
            words_in_s = re.findall(r"\b[A-Za-z]{6,15}\b", s_clean)
        if not words_in_s:
            continue
        key = random.choice(words_in_s)
        key_l = key.strip()
        # construct question by masking the key
        masked = re.sub(re.escape(key_l), "_____", s_clean, flags=re.IGNORECASE)
        if masked == s_clean:
            continue
        # choose distractors
        distractors = []
        pool = [w for w in candidates if w.lower() != key_l.lower()]
        random.shuffle(pool)
        for p in pool[:10]:
            if p.lower() != key_l.lower():
                distractors.append(p.capitalize())
            if len(distractors) >= 3:
                break
        if len(distractors) < 3:
            # fallback: generate random variations
            filler = ["concept", "method", "process", "component"]
            while len(distractors) < 3:
                distractors.append(random.choice(filler))
        options = distractors[:3] + [key_l.capitalize()]
        random.shuffle(options)
        answer_text = next((opt for opt in options if opt.lower() == key_l.lower() or key_l.lower() in opt.lower()), options[-1])
        qobj = {
            "question": masked,
            "options": options,
            "answer": answer_text,
            "source": "offline-fallback"
        }
        # avoid duplicate questions
        if masked.lower() in used_sentences:
            continue
        used_sentences.add(masked.lower())
        questions.append(qobj)
        i += 1

    # If not enough, synthesize trivial definitional questions from top keywords
    idx = 0
    while len(questions) < target_n and idx < len(candidates):
        kw = candidates[idx]
        idx += 1
        q = f"What best describes '{kw.capitalize()}' as used in the text?"
        # create options using other top keywords
        other = [c.capitalize() for c in candidates if c != kw][:3]
        while len(other) < 3:
            other.append("A concept")
        options = [other[0], other[1], other[2], kw.capitalize()]
        random.shuffle(options)
        answer_text = kw.capitalize()
        questions.append({"question": q, "options": options, "answer": answer_text, "source": "offline-fallback"})
    # ensure length exactly target_n
    return questions[:target_n]

# ---------------------------
# GeminiQuiz class (free-tier: gemini-1.5-flash)
# ---------------------------
class GeminiQuiz:
    def __init__(self, api_key: str, api_base: str = None, model: str = None, temperature: float = 0.45):
        self.api_key = api_key
        self.api_base = api_base or API_BASE_DEFAULT
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature

    def _build_prompt(self, text: str, difficulty: str):
        # Keep prompt explicit, JSON-first; request exact structure
        diff_text = (
            "Easy: create direct recall questions, definitions, short fact checks."
            if difficulty == "easy"
            else "Hard: create reasoning, inference, comparison, or multi-step logic questions (avoid verbatim recall)."
        )

        prompt = f"""
You are an expert educational quiz generator.

{diff_text}

Generate exactly 20 multiple-choice questions (MCQs) from the text below.
Each question object must contain:
- question: the question text
- options: a list of four answer strings (A-D)
- answer: the exact text of the correct option (must match one of options)
- source: an optional short source string

Return output strictly as a JSON array (no extra commentary, no markdown, no explanation). Example of each item:
{{
  "question": "Question text ...",
  "options": ["Option A", "Option B", "Option C", "Option D"],
  "answer": "Option B",
  "source": "Section name"
}}

Text:
{text}
"""
        # strip and limit accidentally huge text (server tokens)
        return prompt.strip()

    def _call_api(self, prompt: str, timeout: int = 90):
        url = f"{self.api_base}/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(self.temperature),
                "maxOutputTokens": int(MAX_OUTPUT_TOKENS)
            }
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        # raise_for_status will allow us to inspect JSON error messages via exception handlers
        resp.raise_for_status()
        return resp.json()

    def generate_quiz(self, text: str, difficulty: str = "easy", retries: int = RETRIES):
        prompt = self._build_prompt(text, difficulty)
        last_err = None
        for attempt in range(1, retries + 1):
            try:
                resp_json = self._call_api(prompt, timeout=120)
                # Try to extract model text
                model_text = deep_find_text(resp_json) or ""
                model_text = clean_text(model_text)
                if not model_text:
                    last_err = "Empty response text"
                    time.sleep(RETRY_BACKOFF * attempt)
                    continue
                # first attempt: parse as JSON substring
                quiz = self._parse_json_like_text(model_text)
                if quiz and len(quiz) >= 20:
                    # normalize to exactly 20
                    return quiz[:20]
                # if returned fewer questions, try fallback parsing heuristics
                quiz2 = self._fallback_parse_blocks(model_text)
                if quiz2 and len(quiz2) >= 20:
                    return quiz2[:20]
                # keep what we have if at least some questions; else retry
                if quiz2 and len(quiz2) > 0:
                    # try one more attempt but can accept partial if API is flaky (still prefer full)
                    if attempt == retries:
                        # pad using offline generator if <20
                        needed = 20 - len(quiz2)
                        pad = offline_generate_quiz_from_text(text, difficulty, needed)
                        return (quiz2 + pad)[:20]
                # else retry
                last_err = "Parsed response did not contain 20 valid questions"
                time.sleep(RETRY_BACKOFF * attempt)
            except requests.HTTPError as he:
                # attempt to read server error message
                try:
                    err_json = he.response.json()
                    msg = err_json.get("error", {}).get("message", str(err_json))
                except Exception:
                    msg = str(he)
                # if quota error or similar, propagate as last_err and break / fallback
                last_err = f"HTTPError: {msg}"
                # if quota-like error, don't spam retries - break to fallback
                if "quota" in msg.lower() or "billing" in msg.lower():
                    break
                time.sleep(RETRY_BACKOFF * attempt)
            except Exception as e:
                last_err = f"Exception: {e}"
                time.sleep(RETRY_BACKOFF * attempt)

        # As final fallback, generate offline
        st.warning("Gemini API unavailable or failed. Using offline fallback generator.")
        fallback = offline_generate_quiz_from_text(text, difficulty, target_n=20)
        return fallback

    def _parse_json_like_text(self, text: str):
        # Attempt to extract a JSON array substring and parse
        try:
            first = text.index("[")
            last = text.rindex("]")
            json_sub = text[first:last+1]
            parsed = json.loads(json_sub)
            questions = []
            if isinstance(parsed, list):
                for item in parsed:
                    if not isinstance(item, dict):
                        continue
                    q = item.get("question") or item.get("q")
                    options = item.get("options") or item.get("choices") or item.get("opts")
                    ans = item.get("answer") or item.get("correct")
                    src = item.get("source") or item.get("section") or ""
                    if q and isinstance(options, list) and len(options) >= 2 and ans:
                        # ensure options are strings
                        options = [str(o).strip() for o in options][:4]
                        # ensure answer matches an option if possible
                        if ans not in options and isinstance(ans, str) and len(ans.strip()) == 1:
                            idx = ord(ans.strip().upper()) - 65
                            if 0 <= idx < len(options):
                                ans = options[idx]
                        questions.append({"question": str(q).strip(), "options": options, "answer": str(ans).strip(), "source": str(src)})
            return questions
        except Exception:
            return None

    def _fallback_parse_blocks(self, text: str):
        # Heuristic parse similar to older logic
        questions = []
        blocks = re.split(r'\n\s*\n', text)
        for block in blocks:
            if len(questions) >= 20:
                break
            if "Q:" in block or "Question" in block:
                q = ""
                options = []
                ans = ""
                # extract Q:
                if "Q:" in block:
                    q = block.split("Q:",1)[1].splitlines()[0].strip()
                else:
                    # find line starting with "Question"
                    m = re.search(r'Question[:\s]*\d*[:\s]*(.*)', block, re.IGNORECASE)
                    if m:
                        q = m.group(1).strip()
                # extract options lines
                for letter in ["A)", "A.", "A "]:
                    if letter in block:
                        # find A) ... D)
                        opts = re.findall(r'(?:A\)|A\.|A\s)(.*?)\n(?:B\)|B\.|B\s)', block, flags=re.S)
                        # fallback simpler capture
                        break
                # simpler approach: find lines starting with A/B/C/D
                for line in block.splitlines():
                    line = line.strip()
                    if re.match(r'^[ABCD][\)\.\:]\s*', line):
                        opt_text = re.split(r'^[ABCD][\)\.\:]\s*', line, maxsplit=1)[1].strip()
                        options.append(opt_text)
                # answer
                m = re.search(r'Answer[:\s]*([A-D]|\w.+)', block, re.IGNORECASE)
                if m:
                    ans_line = m.group(1).strip()
                    # if single letter, map
                    if len(ans_line) == 1 and ans_line.upper() in "ABCD" and len(options) >= 1:
                        idx = ord(ans_line.upper()) - 65
                        if 0 <= idx < len(options):
                            ans = options[idx]
                    else:
                        ans = ans_line
                if q and options:
                    if not ans:
                        ans = options[0]
                    questions.append({"question": q, "options": options[:4], "answer": ans, "source": ""})
        return questions

# ---------------------------
# Render Single Question Card (preserve look)
# ---------------------------
def render_question(card, index, total, dark_mode):
    bg = "#1E1E1E" if dark_mode else "#F9F9F9"
    text = "#FFFFFF" if dark_mode else "#000000"
    sub = "#AAAAAA" if dark_mode else "#555555"

    # escape content to avoid raw HTML injections
    title = escape(card.get("title", card.get("question", f"Q{index+1}")))
    question_text = escape(card.get("question", ""))
    options = card.get("options", [])
    source = escape(card.get("source", "N/A"))
    answer = escape(card.get("answer", "N/A"))

    st.markdown(
        f"""
        <div style="height:95vh; display:flex; flex-direction:column; justify-content:space-between;
                    background-color:{bg}; color:{text}; padding:3rem; border-radius:12px;
                    box-shadow:0 0 10px rgba(0,0,0,0.15); scroll-snap-align:start;">
            <div>
                <h3 style="margin-top:0;">Q{index + 1}. {question_text}</h3>
                <ul style="list-style-type:none; padding-left:0;">
                    {''.join(f"<li style='margin:0.6em 0;'>({chr(65+i)}) {escape(opt)}</li>" for i,opt in enumerate(options))}
                </ul>
            </div>
        """, unsafe_allow_html=True
    )

    with st.expander("Reveal Answer"):
        st.success(f"Answer: {answer}")

    st.markdown(
        f"""
            <div style="font-size:0.8em; color:{sub}; text-align:right; border-top:1px solid {sub}; padding-top:0.5em;">
                Source: {source}
            </div>
        </div>
        """, unsafe_allow_html=True
    )
    st.progress((index + 1) / total, text=f"Question {index + 1} of {total}")

# ---------------------------
# Streamlit page (main)
# ---------------------------
def main():
    st.set_page_config(page_title="Quiz Me", layout="wide")
    st.markdown(
        """
        <style>
        html, body, [class*="stAppViewContainer"] {
            scroll-snap-type: y mandatory;
            overflow-y: scroll;
        }
        </style>
        """, unsafe_allow_html=True
    )

    gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    gemini_base = st.secrets.get("GEMINI_API_BASE") or os.getenv("GEMINI_API_BASE") or API_BASE_DEFAULT
    gemini_model = DEFAULT_MODEL  # enforce free-tier model

    if not gemini_key:
        st.error("Missing GEMINI_API_KEY. Please set it in Streamlit secrets or environment variables.")
        st.stop()

    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False
    if "quiz_difficulty" not in st.session_state:
        st.session_state.quiz_difficulty = "easy"

    # Header
    left, right = st.columns([6, 1])
    with left:
        st.title("Quiz Me")
    with right:
        icon = "🌙" if not st.session_state.dark_mode else "☀️"
        if st.button(icon, key="quiz_theme_toggle"):
            st.session_state.dark_mode = not st.session_state.dark_mode

    # Sidebar controls
    with st.sidebar:
        st.subheader("Quiz Settings")
        diff = st.radio("Difficulty", ["easy", "hard"], index=["easy", "hard"].index(st.session_state.quiz_difficulty))
        st.session_state.quiz_difficulty = diff
        shuffle_q = st.checkbox("Shuffle questions")

        if st.button("Generate Quiz"):
            if "conversation" not in st.session_state or st.session_state.conversation is None:
                st.warning("Please analyze a document first from the main page.")
                st.stop()

            retriever = st.session_state.conversation.retriever
            # try to gather more document coverage
            docs = [d.page_content for d in retriever.vectorstore.docstore._dict.values()]
            combined_text = "\n".join(docs[:10]) if docs else ""

            if not combined_text.strip():
                st.warning("No extracted text found from the retriever. Provide document or URL in main page.")
                st.stop()

            gem = GeminiQuiz(api_key=gemini_key, api_base=gemini_base, model=gemini_model)
            with st.spinner("Generating quiz..."):
                qs = gem.generate_quiz(combined_text, difficulty=diff, retries=RETRIES)
            if not qs or len(qs) < 1:
                st.error("No questions generated. A fallback will be used.")
                qs = offline_generate_quiz_from_text(combined_text, diff, target_n=20)
            if shuffle_q:
                random.shuffle(qs)
            st.session_state.quiz_questions = qs
            st.success(f"Generated {len(st.session_state.quiz_questions)} questions.")

    qs = st.session_state.quiz_questions
    if not qs:
        st.info("No quiz yet. Generate from sidebar.")
        return

    for idx, q in enumerate(qs):
        render_question(q, idx, len(qs), st.session_state.dark_mode)
        st.markdown("<br><br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
