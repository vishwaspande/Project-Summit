import streamlit as st

st.title("🇮🇳 Streamlit Test")
st.write("✅ If you see this, Streamlit is working!")

# Test imports
try:
    import anthropic
    st.success("✅ Anthropic library imported")
except:
    st.error("❌ Anthropic library missing")

try:
    import yfinance
    st.success("✅ yfinance library imported")
except:
    st.error("❌ yfinance library missing")

# Test API key
import os
api_key = os.getenv('ANTHROPIC_API_KEY')
if api_key:
    st.success(f"✅ API key found: {api_key[:15]}...")
else:
    st.error("❌ API key not found")

# Test if our main file exists
if os.path.exists('indian_stock_market_agent.py'):
    st.success("✅ indian_stock_market_agent.py file exists")
    
    # Try importing it
    try:
        from indian_stock_market_agent import IndianStockMarketAgent
        st.success("✅ IndianStockMarketAgent imported successfully")
    except Exception as e:
        st.error(f"❌ Error importing IndianStockMarketAgent: {e}")
else:
    st.error("❌ indian_stock_market_agent.py file not found")