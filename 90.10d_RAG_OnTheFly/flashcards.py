import streamlit as st
import random
import os
import requests
import re
# -------------------------------------------------------------------
# Gemini wrapper (modular and independent)
# -------------------------------------------------------------------
class GeminiChat:
    def __init__(self, api_key, api_base=None, model="models/gemini-2.5-flash",
                 temperature=0.4, max_output_tokens=2048):
        self.api_key = api_key
        self.api_base = api_base or "https://generativelanguage.googleapis.com/v1beta"
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def generate_flashcards(self, text, difficulty="detailed"):
        prompt = f"""
You are an expert educational summarizer and flashcard generator.

Generate a full set of study flashcards from the following material.
Each flashcard should cover one key idea in full detail and fill roughly one screen of content.

Difficulty: {difficulty}

Output at least 5 flashcards.

Each flashcard must be formatted EXACTLY like this — plain text only:
Flashcard:
Title: ...
Points:
- ...
- ...
Example: ...
Keywords: ...
Source: ...

Text:
{text}
        """.strip()

        url = f"{self.api_base}/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": float(self.temperature),
                "maxOutputTokens": int(self.max_output_tokens),
            },
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=90)
            data = response.json()
        except Exception as e:
            st.error(f"Gemini connection error: {e}")
            return []

        # --- Robust parsing logic ---
        try:
            if response.status_code != 200:
                err_msg = data.get("error", {}).get("message", str(data))
                st.error(f"Gemini API error ({response.status_code}): {err_msg}")
                return []

            # Safely extract text from the candidates
            text_out = None
            if "candidates" in data and data["candidates"]:
                c = data["candidates"][0]
                if "content" in c and "parts" in c["content"]:
                    # Normal case
                    parts = c["content"]["parts"]
                    if isinstance(parts, list) and len(parts) > 0:
                        text_out = parts[0].get("text", "")
                elif "content" in c and isinstance(c["content"], dict):
                    # Alternate structure
                    text_out = c["content"].get("text", "")
                elif "output" in c:
                    # Sometimes Gemini returns 'output' field
                    text_out = c["output"]

            if not text_out:
                st.error("Gemini returned an empty or unexpected response structure.")
                st.json(data)  # helpful for debugging
                return []

            return self.parse_flashcards(text_out)

        except Exception as e:
            st.error(f"Gemini response parsing failed: {e}")
            st.json(data)
            return []

    def parse_flashcards(self, text):
        cards = []
        blocks = text.split("Flashcard:")
        for block in blocks:
            block = block.strip()
            if not block:
                continue
            title = self._extract(block, "Title:")
            points = re.findall(r"^- (.*)", block, flags=re.MULTILINE)
            example = self._extract(block, "Example:")
            keywords = self._extract(block, "Keywords:")
            source = self._extract(block, "Source:")
            cards.append({
                "title": title or "Untitled",
                "points": points,
                "example": example,
                "keywords": keywords,
                "source": source,
            })
        return cards

    def _extract(self, text, key):
        match = re.search(rf"{key}\s*(.*)", text)
        return match.group(1).strip() if match else ""


# -------------------------------------------------------------------
# Render one flashcard
# -------------------------------------------------------------------
def render_flashcard(card, index, total, dark_mode):
    bg_color = "#1E1E1E" if dark_mode else "#FAFAFA"
    text_color = "#FFFFFF" if dark_mode else "#111111"
    sub_color = "#AAAAAA" if dark_mode else "#666666"

    # Clean text fields and strip HTML leftovers
    title = clean_html(card.get("title", ""))
    example = clean_html(card.get("example", ""))
    keywords = clean_html(card.get("keywords", ""))
    source = clean_html(card.get("source", ""))

    # Avoid rendering HTML fragments from Gemini
    # Remove accidental "</div>" or stray tags that appear as text
    for var in [title, example, keywords, source]:
        var = re.sub(r"</?div[^>]*>", "", var, flags=re.IGNORECASE)
        var = var.strip()

    # Render card body
    st.markdown(
        f"""
        <div style="
            height:95vh;
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            background-color:{bg_color};
            color:{text_color};
            padding:3rem 4rem;
            border-radius:16px;
            box-shadow:0 0 14px rgba(0,0,0,0.15);
            scroll-snap-align:start;
        ">
            <div style="flex-grow:1;">
                <h2 style="margin-bottom:1rem;">{title}</h2>
                <ul style="line-height:1.6; font-size:1.05rem;">
                    {''.join(f'<li>{clean_html(p)}</li>' for p in card.get('points', []))}
                </ul>
                {f"<p><b>Example:</b> {example}</p>" if example else ""}
                {f"<p style='color:{sub_color}; font-size:0.8em;'>Keywords: {keywords}</p>" if keywords else ""}
            </div>
            {f'<div style="font-size:0.7em; color:{sub_color}; text-align:right; border-top:1px solid {sub_color}; padding-top:0.3em; margin-top:1.5em;">Source: {source}</div>' if source else ''}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress((index + 1) / total, text=f"Card {index + 1} of {total}")
def clean_html(raw_text):
    """Remove stray HTML tags that Gemini might output."""
    return re.sub(r"<[^>]+>", "", raw_text or "").strip()


def render_flashcard(card, index, total, dark_mode):
    bg_color = "#1E1E1E" if dark_mode else "#FAFAFA"
    text_color = "#FFFFFF" if dark_mode else "#111111"
    sub_color = "#AAAAAA" if dark_mode else "#666666"

    # Clean text fields
    title = clean_html(card["title"])
    example = clean_html(card["example"])
    keywords = clean_html(card["keywords"])
    source = clean_html(card["source"])

    st.markdown(
        f"""
        <div style="
            height:95vh;
            display:flex;
            flex-direction:column;
            justify-content:space-between;
            background-color:{bg_color};
            color:{text_color};
            padding:3rem 4rem;
            border-radius:16px;
            box-shadow:0 0 14px rgba(0,0,0,0.15);
            scroll-snap-align:start;
        ">
            <div style="flex-grow:1;">
                <h2 style="margin-bottom:1rem;">{title}</h2>
                <ul style="line-height:1.6; font-size:1.05rem;">
                    {''.join(f'<li>{clean_html(p)}</li>' for p in card['points'])}
                </ul>
                {f"<p><b>Example:</b> {example}</p>" if example else ""}
                {f"<p style='color:{sub_color}; font-size:0.8em;'>Keywords: {keywords}</p>" if keywords else ""}
            </div>
            <div style="font-size:0.7em; color:{sub_color}; text-align:right; border-top:1px solid {sub_color}; padding-top:0.3em; margin-top:1.5em;">
                Source: {source or 'N/A'}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress((index + 1) / total, text=f"Card {index + 1} of {total}")

# -------------------------------------------------------------------
# Streamlit Page
# -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Flashcards", layout="wide")

    # CSS for smooth scroll
    st.markdown(
        """
        <style>
        html, body, [class*="stAppViewContainer"] {
            scroll-snap-type: y mandatory;
            overflow-y: scroll;
            scroll-behavior: smooth;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    gemini_key = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")
    gemini_base = st.secrets.get("GEMINI_API_BASE") or os.getenv("GEMINI_API_BASE")

    if not gemini_key:
        st.error("Missing GEMINI_API_KEY.")
        st.stop()

    # Session init
    st.session_state.setdefault("flashcards", [])
    st.session_state.setdefault("dark_mode", False)
    st.session_state.setdefault("difficulty", "detailed")

    # Header row
    c1, c2 = st.columns([6, 1])
    with c1:
        st.title("Flashcards")
    with c2:
        icon = "🌙" if not st.session_state.dark_mode else "☀️"
        if st.button(icon, key="theme"):
            st.session_state.dark_mode = not st.session_state.dark_mode

    # Sidebar controls
    with st.sidebar:
        st.subheader("Flashcard Settings")
        diff = st.radio(
            "Difficulty Level",
            ["simple", "detailed", "exam-style"],
            index=["simple", "detailed", "exam-style"].index(st.session_state.difficulty),
        )
        st.session_state.difficulty = diff
        shuffle = st.checkbox("Shuffle order")

        if st.button("Generate Flashcards"):
            if "conversation" not in st.session_state or st.session_state.conversation is None:
                st.warning("Please analyze a document first.")
                st.stop()

            st.info("Fetching from retriever...")
            retriever = st.session_state.conversation.retriever
            all_docs = [d.page_content for d in retriever.vectorstore.docstore._dict.values()]
            sample_size = min(10, len(all_docs))
            combined_text = "\n".join(random.sample(all_docs, sample_size))
            gem = GeminiChat(gemini_key, gemini_base)
            with st.spinner("Generating detailed flashcards..."):
                cards = gem.generate_flashcards(combined_text, difficulty=diff)
                if shuffle:
                    random.shuffle(cards)
                st.session_state.flashcards = cards
            st.success(f"Generated {len(cards)} flashcards.")

    # Display flashcards
    cards = st.session_state.flashcards
    if not cards:
        st.info("No flashcards yet. Generate them using the sidebar.")
        return

    for i, card in enumerate(cards):
        render_flashcard(card, i, len(cards), st.session_state.dark_mode)
        st.markdown("<br><br>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
