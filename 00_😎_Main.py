import streamlit as st
import requests
import re
import random
from collections import defaultdict
import math

# =========================
# Data sources
# =========================
DATASOURCE = {
    "Ramayana (English)": "https://www.gutenberg.org/files/24869/24869-0.txt",
    "Mahabharata (English)": "https://www.gutenberg.org/files/15474/15474-0.txt",
    "Bhagavad Gita (English - Arnold)": "https://www.gutenberg.org/ebooks/2388.txt.utf-8", # Direct text
    # Or use the main page, which also works:
    # "Bhagavad Gita (English - Arnold)": "https://www.gutenberg.org/ebooks/2388",
}



# =========================
# Load & preprocess text
# =========================
@st.cache_data
def load_text(url):
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        text = response.text.lower()
        # Keep basic punctuation for better learning
        text = re.sub(r"[^a-z\s.,!?']", " ", text)
        text = re.sub(r"\s+", " ", text)
        words = text.strip().split()
        return words
    except Exception as e:
        st.error(f"Error loading text: {e}")
        return None

# =========================
# Build Order-3 Markov Chain (TEACHING FOCUS)
# =========================
def build_markov_chain_order3(words):
    """Build a 3-word context Markov model with teaching visualization"""
    model = defaultdict(lambda: defaultdict(int))
    
    st.write("### 🔍 Building Order-3 Model")
    st.write(f"Processing {len(words)} words...")
    
    # Count transitions for teaching statistics
    total_transitions = 0
    context_examples = []
    
    for i in range(len(words) - 3):
        context = (words[i], words[i + 1], words[i + 2])
        next_word = words[i + 3]
        model[context][next_word] += 1
        total_transitions += 1
        
        # Collect examples for teaching
        if len(context_examples) < 5 and context not in [c[0] for c in context_examples]:
            context_examples.append((context, next_word))
    
    # Convert counts to probabilities
    prob_model = {}
    for context, next_words in model.items():
        total = sum(next_words.values())
        prob_model[context] = {
            w: c / total for w, c in next_words.items()
        }
    
    # Teaching display
    st.success(f"✅ Model built with {len(model):,} unique 3-word contexts")
    st.info(f"📊 {total_transitions:,} total transitions learned")
    
    # Show examples for teaching
    st.write("### 🧪 Learning Examples")
    st.write("The model learned patterns like:")
    for i, (context, next_word) in enumerate(context_examples[:3]):
        context_str = " ".join(context)
        st.code(f'Context: "{context_str}" → Next: "{next_word}"')
    
    return prob_model

# =========================
# Temperature scaling (TEACHING VERSION)
# =========================
def apply_temperature_with_explanation(probs, temperature):
    """Apply temperature with visual explanation"""
    if temperature <= 0.01:
        # Deterministic choice - teaching mode
        max_idx = probs.index(max(probs))
        result = [0.0] * len(probs)
        result[max_idx] = 1.0
        return result, "Deterministic (always picks highest probability)"
    
    # Apply temperature
    scaled = []
    explanation = []
    
    for p in probs:
        # Avoid log(0)
        p_safe = max(p, 1e-10)
        # Apply temperature
        scaled_p = math.exp(math.log(p_safe) / temperature)
        scaled.append(scaled_p)
        explanation.append(f"{p:.3f} → {scaled_p:.3f}")
    
    total = sum(scaled)
    normalized = [p / total for p in scaled]
    
    # Generate explanation
    expl_text = f"Temperature {temperature:.1f} applied:\n"
    expl_text += f"Before: {[f'{p:.3f}' for p in probs]}\n"
    expl_text += f"After:  {[f'{p:.3f}' for p in normalized]}"
    
    return normalized, expl_text

# =========================
# Find best context for starting phrase
# =========================
def find_starting_context(phrase, model):
    """Find the best context for a starting phrase (TEACHING FOCUS)"""
    words = phrase.lower().strip().split()
    
    if len(words) < 3:
        st.warning(f"⚠️  Starting phrase '{phrase}' has {len(words)} words")
        st.write("For Order-3 model, we need at least 3 words for context.")
        st.write("Please enter at least 3 words, or we'll use a random start.")
        return None
    
    # Try exact 3-word match from end
    context = tuple(words[-3:])
    if context in model:
        st.success(f"✅ Perfect match found: {' '.join(context)}")
        return context
    
    # Teaching: Show why we didn't find exact match
    st.warning(f"⚠️  Exact context {' '.join(context)} not found in training data")
    
    # Try to find any matching 3-gram in the phrase
    for i in range(len(words) - 3, -1, -1):
        test_context = tuple(words[i:i+3])
        if test_context in model:
            st.info(f"📌 Using alternative context from phrase: {' '.join(test_context)}")
            return test_context
    
    # No match found in phrase
    st.warning("🔍 No 3-word sequence from your phrase found in training data")
    st.write("Using a random context from the model instead.")
    return None

# =========================
# Generate text with enhanced teaching explanations
# =========================
def generate_text_with_teaching(model, start_phrase, n_words, temperature):
    """Generate text with step-by-step teaching explanations"""
    
    # Get all contexts for fallback
    contexts = list(model.keys())
    
    # Find starting context
    start_context = find_starting_context(start_phrase, model)
    
    if start_context is None:
        st.info("🎲 Selecting random starting context from model...")
        start_context = random.choice(contexts)
        st.code(f"Random start: {' '.join(start_context)}")
    else:
        # Verify we start with user's phrase
        start_phrase_words = start_phrase.lower().strip().split()
        last_three = ' '.join(start_phrase_words[-3:])
        if ' '.join(start_context) != last_three:
            st.info(f"📝 Using: {' '.join(start_context)} (from your phrase)")
    
    w1, w2, w3 = start_context
    generated = [w1, w2, w3]
    steps = []
    
    # Teaching: Show initial state
    st.write("---")
    st.write("### 🔄 Generation Process")
    
    # Generate words
    for step in range(n_words):
        context = (w1, w2, w3)
        
        # Handle unknown context
        if context not in model:
            st.warning(f"⚠️  Context '{w1} {w2} {w3}' not found in model!")
            st.write("This happens when the specific 3-word sequence wasn't in the training data.")
            st.write("Falling back to random context...")
            context = random.choice(contexts)
            w1, w2, w3 = context
        
        # Get candidates and probabilities
        candidates = list(model[context].items())
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Teaching: Show top candidates
        words, probs = zip(*candidates)
        
        # Apply temperature with explanation
        adjusted_probs, temp_explanation = apply_temperature_with_explanation(probs, temperature)
        
        # Choose next word
        chosen_idx = random.choices(range(len(words)), adjusted_probs)[0]
        chosen_word = words[chosen_idx]
        
        # Store step information for teaching
        step_info = {
            "context": f"{w1} {w2} {w3}",
            "top_candidates": list(zip(words[:5], probs[:5], adjusted_probs[:5])),
            "chosen": chosen_word,
            "chosen_prob": probs[chosen_idx],
            "temperature_explanation": temp_explanation,
            "step": step + 1
        }
        steps.append(step_info)
        
        # Update generated text and context
        generated.append(chosen_word)
        w1, w2, w3 = w2, w3, chosen_word
    
    return " ".join(generated), steps

# =========================
# Compare Order-2 vs Order-3 (TEACHING FOCUS)
# =========================
def build_comparison_models(words):
    """Build both Order-2 and Order-3 models for teaching comparison"""
    st.write("## 🔬 Model Comparison: Order-2 vs Order-3")
    
    # Build Order-2 model
    st.write("### 📐 Building Order-2 Model")
    model_order2 = defaultdict(lambda: defaultdict(int))
    for i in range(len(words) - 2):
        context = (words[i], words[i + 1])
        next_word = words[i + 2]
        model_order2[context][next_word] += 1
    
    # Convert to probabilities
    prob_model2 = {}
    for context, next_words in model_order2.items():
        total = sum(next_words.values())
        prob_model2[context] = {w: c/total for w, c in next_words.items()}
    
    # Build Order-3 model
    st.write("### 📏 Building Order-3 Model")
    model_order3 = defaultdict(lambda: defaultdict(int))
    for i in range(len(words) - 3):
        context = (words[i], words[i + 1], words[i + 2])
        next_word = words[i + 3]
        model_order3[context][next_word] += 1
    
    # Convert to probabilities
    prob_model3 = {}
    for context, next_words in model_order3.items():
        total = sum(next_words.values())
        prob_model3[context] = {w: c/total for w, c in next_words.items()}
    
    # Teaching comparison
    st.success("✅ Both models built successfully!")
    
    comparison_data = {
        "order2": {
            "name": "Order-2 (2-word context)",
            "contexts": len(prob_model2),
            "example": next(iter(prob_model2.items())) if prob_model2 else None,
            "model": prob_model2
        },
        "order3": {
            "name": "Order-3 (3-word context)",
            "contexts": len(prob_model3),
            "example": next(iter(prob_model3.items())) if prob_model3 else None,
            "model": prob_model3
        }
    }
    
    # Display comparison
    st.write("### 📊 Model Statistics")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Order-2 Contexts", f"{comparison_data['order2']['contexts']:,}")
        if comparison_data['order2']['example']:
            ctx, probs = comparison_data['order2']['example']
            st.caption(f"Example: {' '.join(ctx)} → {list(probs.keys())[0]} ({list(probs.values())[0]:.2%})")
    
    with col2:
        st.metric("Order-3 Contexts", f"{comparison_data['order3']['contexts']:,}")
        if comparison_data['order3']['example']:
            ctx, probs = comparison_data['order3']['example']
            st.caption(f"Example: {' '.join(ctx)} → {list(probs.keys())[0]} ({list(probs.values())[0]:.2%})")
    
    # Teaching insight
    st.info("""
    **🎓 Teaching Insight**: 
    - **Order-2**: Simpler, faster, but less context
    - **Order-3**: More accurate, better coherence, but needs more data
    - Each additional word in context exponentially increases prediction quality
    """)
    
    return comparison_data

# =========================
# Streamlit UI - TEACHING FOCUS
# =========================
st.set_page_config(page_title="Markov Model Teaching Lab", layout="wide")
st.title("🎓 Markov Model Teaching Lab: Order-3 Edition")
st.caption("Learn how language models work by building a 3-word context Markov chain")

# Sidebar controls
st.sidebar.header("🎛️ Controls")

book = st.sidebar.selectbox("Select Training Text", DATASOURCE.keys())
start_phrase = st.sidebar.text_input(
    "Starting phrase (min 3 words)", 
    value="lord rama said to",
    help="Enter at least 3 words for Order-3 context"
)
num_words = st.sidebar.slider("Words to generate", 10, 100, 30)
temperature = st.sidebar.slider(
    "Temperature", 0.0, 2.0, 1.0, 0.1,
    help="0 = deterministic, 1 = original probabilities, 2 = more random"
)

# Teaching mode toggle
teaching_mode = st.sidebar.checkbox("📚 Detailed Teaching Mode", value=True)

# Generate button
if st.sidebar.button("🚀 Generate & Learn"):
    
    # Load text
    with st.spinner("📥 Loading text..."):
        words = load_text(DATASOURCE[book])
        if words is None:
            st.error("Failed to load text. Please try again.")
            st.stop()
    
    # Build and compare models
    comparison = build_comparison_models(words)
    
    st.write("---")
    st.write("## 🎯 Order-3 Generation (Main Model)")
    
    # Generate text with Order-3 model
    with st.spinner("🧠 Generating text with Order-3 model..."):
        text, steps = generate_text_with_teaching(
            comparison["order3"]["model"], 
            start_phrase, 
            num_words, 
            temperature
        )
    
    # Display results
    st.subheader("📝 Generated Text (Order-3)")
    st.markdown(f'<div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4CAF50;">{text}</div>', 
                unsafe_allow_html=True)
    
    # Display step-by-step teaching
    if teaching_mode and steps:
        st.write("---")
        st.subheader("🔍 Step-by-Step Teaching")
        
        # Let user navigate through steps
        step_to_show = st.selectbox(
            "Select step to examine",
            range(1, min(11, len(steps) + 1)),
            format_func=lambda x: f"Step {x}: Context → Word"
        )
        
        if step_to_show <= len(steps):
            step = steps[step_to_show - 1]
            
            st.write(f"### Step {step_to_show}")
            
            # Context box
            st.markdown(f"""
            **Previous 3-word context:**  
            `{step['context']}`
            """)
            
            # Top candidates
            st.write("**Top 5 candidate next words:**")
            for word, orig_prob, adj_prob in step["top_candidates"][:5]:
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.write(f"`{word}`")
                with col2:
                    st.write(f"Orig: {orig_prob:.3f}")
                with col3:
                    st.write(f"Adj: {adj_prob:.3f}")
            
            # Temperature explanation
            if temperature != 1.0:
                with st.expander("🧪 Temperature Effect"):
                    st.text(step["temperature_explanation"])
            
            # Chosen word
            st.success(f"""
            **✅ Selected word:** `{step['chosen']}`  
            **Probability:** {step['chosen_prob']:.3f}
            """)
            
            # Teaching insight
            st.info(f"""
            **🎓 Teaching Moment:**  
            The model looked at the last 3 words `{step['context']}` and chose 
            `{step['chosen']}` as the most likely next word based on patterns 
            learned from the training text.
            """)
    
    # Show simple comparison with Order-2
    st.write("---")
    st.subheader("⚖️ Quick Comparison: Order-2 vs Order-3")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Order-2 Output**")
        # Generate quick Order-2 sample
        model2 = comparison["order2"]["model"]
        if start_phrase:
            start_words = tuple(start_phrase.lower().split()[-2:])
            if start_words in model2:
                # Simple Order-2 generation
                generated = list(start_words)
                w1, w2 = start_words
                for _ in range(10):  # Just first 10 words for comparison
                    context = (w1, w2)
                    if context in model2:
                        candidates = list(model2[context].items())
                        if candidates:
                            chosen = random.choices(
                                [w for w, _ in candidates],
                                [p for _, p in candidates]
                            )[0]
                            generated.append(chosen)
                            w1, w2 = w2, chosen
                
                st.code(' '.join(generated[:15]) + "...")
    
    with col2:
        st.write("**Order-3 Output**")
        st.code(text[:100] + "...")
    
    # Final teaching summary
    st.write("---")
    st.subheader("📚 Learning Summary")
    
    st.markdown("""
    ### What you learned today:
    
    1. **Order-3 Markov Models** use 3-word contexts for prediction
    2. **Context matters**: More context = better coherence
    3. **Temperature control**: Adjusts randomness of predictions
    4. **Training data**: The model learns from patterns in the text
    5. **Limitations**: Can't generate truly novel ideas, only recombine learned patterns
    
    ### Try experimenting with:
    - Different starting phrases
    - Temperature = 0 (deterministic) vs 2 (very creative)
    - Different source texts
    """)

# Initial teaching content
else:
    st.write("## 🎓 Welcome to the Markov Model Teaching Lab!")
    
    st.markdown("""
    ### 👨‍🏫 What you'll learn:
    
    **1. What is a Markov Chain?**
    - A simple statistical model that predicts the next item based only on the current state
    - In language: predicts next word based on previous words
    
    **2. Order-3 vs Order-2: What's the difference?**
    
    | Order | Context | Example | Accuracy |
    |-------|---------|---------|----------|
    | Order-2 | Last 2 words | "rama went" → "to" | Basic |
    | Order-3 | Last 3 words | "lord rama went" → "to" | Better! |
    
    **3. Key Concepts:**
    - **Context Window**: How many previous words the model considers
    - **Probability Distribution**: For each context, a list of possible next words with probabilities
    - **Temperature**: Controls randomness vs determinism
    
    ### 🎯 Today's Goal:
    Understand how adding more context (Order-3) improves text generation compared to simpler models.
    
    ### 🚀 Get Started:
    1. Select a training text
    2. Enter a starting phrase (at least 3 words)
    3. Adjust temperature if desired
    4. Click **"Generate & Learn"** in the sidebar
    """)
    
    # Quick example
    st.write("### 💡 Quick Example")
    st.code("""
    Training: "the cat sat on the mat. the cat slept."
    
    Order-2 learns:
    "the cat" → sat (50%), slept (50%)
    
    Order-3 learns:
    "the cat sat" → on (100%)
    "cat sat on" → the (100%)
    "sat on the" → mat (100%)
    
    Better context = better predictions!
    """)
