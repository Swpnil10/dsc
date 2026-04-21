import streamlit as st
import joblib
import re  # We need this built-in library for checking text patterns

# 1. Set up the page layout and title
st.set_page_config(page_title="Anti-Phishing AI", page_icon="🛡️")
st.title("🛡️ Anti-Phishing AI Detector")
st.write("Check emails using our Hybrid Security Engine (Heuristics + AI).")

# 2. Load the trained model and vectorizer
@st.cache_resource
def load_brain():
    model = joblib.load('archive/phishing_model.pkl')
    vectorizer = joblib.load('archive/vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_brain()

# --- NEW: Layer 1 - Rule-Based Sender Check ---
def check_suspicious_sender(email):
    email = email.lower().strip()
    warnings = []
    
    # 1. Check for weird top-level domains often used by spammers
    suspicious_tlds = ['.xyz', '.tk', '.top', '.pw', '.cc']
    if any(email.endswith(tld) for tld in suspicious_tlds):
        warnings.append("Uses a low-trust domain extension (e.g., .xyz, .tk).")
        
    # 2. Check for "Typosquatting" (Faking big brands)
    big_brands = ['paypal', 'microsoft', 'apple', 'google', 'amazon', 'netflix', 'bank']
    for brand in big_brands:
        # If the brand name is in the email, but it doesn't actually end with @brand.com
        if brand in email and not email.endswith(f"@{brand}.com"):
            warnings.append(f"Contains '{brand}' but is not from their official .com address (Possible Spoofing!).")
            
    # 3. Check for excessive numbers in the domain part
    domain_part = email.split('@')[-1] if '@' in email else ""
    if len(re.findall(r'\d', domain_part)) > 3:
        warnings.append("Domain contains an unusually high number of digits.")

    return warnings

# 3. Create the UI Inputs (Now split into two sections)
st.markdown("### 1. Sender Details (Optional)")
sender_email = st.text_input("From:", placeholder="e.g., support@paypal.com")

st.markdown("### 2. Message Content")
user_message = st.text_area("Email Body:", height=200, placeholder="Type or paste the message here...")

# 4. Create the Check button and logic
if st.button("Check Message"):
    
    word_count = len(user_message.split())
    
    if user_message.strip() == "":
        st.warning("Please enter a message body to check!")
        
    elif word_count < 5:
        st.warning(f"⚠️ Message is too short ({word_count} words). Please paste a full email (at least 5 words).")
        
    else:
        st.markdown("---")
        st.markdown("## 🔍 Security Analysis")
        
        # --- NEW: Run Layer 1 (Sender Analysis) ---
        if sender_email != "":
            st.write("#### 🛡️ Layer 1: Sender Check")
            sender_flags = check_suspicious_sender(sender_email)
            
            if sender_flags:
                for flag in sender_flags:
                    st.error(f"🚩 **Suspicious Sender:** {flag}")
            else:
                st.success("✅ Sender address format looks standard.")
            st.write("") # Add a little space

        # --- EXISTING: Run Layer 2 (AI Content Analysis) ---
        st.write("#### 🧠 Layer 2: AI Content Check")
        vectorized_msg = vectorizer.transform([user_message])
        prediction = model.predict(vectorized_msg)[0]
        probability = model.predict_proba(vectorized_msg)[0]
        
        if prediction == 0:
            st.success(f"✅ **LOOKS LEGIT!**")
            st.write(f"**Confidence Score:** {probability[0] * 100:.2f}%")
            st.write("👍 *The message content appears to be safe and professional.*")
            
        elif prediction == 1:
            st.warning(f"🗑️ **SPAM DETECTED!**")
            st.write(f"**Confidence Score:** {probability[1] * 100:.2f}%")
            st.write("📢 *Recommendation: This looks like annoying marketing junk.*")
            
        elif prediction == 2:
            st.error(f"🚨 **PHISHING / SCAM DETECTED!**")
            st.write(f"**Confidence Score:** {probability[2] * 100:.2f}%")
            st.write("⚠️ *Recommendation: Malicious intent detected! Do not click links or share info.*")