# OmniRenovation AI - Phase 1 Pilot

A fast prototype for testing the AI-native renovation platform concept.

## Quick Start (Local)

```bash
# Clone/download this folder
cd omnirenovation-pilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Configuration

You can configure the API key in two ways:

1. **In the app**: Enter your Anthropic API key in the configuration section
2. **Via secrets file**: Create `.streamlit/secrets.toml`:
   ```toml
   ANTHROPIC_API_KEY = "sk-ant-your-key-here"
   ```

## Deploy to Streamlit Cloud (Free)

1. Push this code to a GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repository and `app.py`
5. Add your `ANTHROPIC_API_KEY` in the Secrets section
6. Deploy!

Your app will be live at `https://your-app-name.streamlit.app`

## Deploy to Railway ($5/month)

1. Create account at [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Add environment variable: `ANTHROPIC_API_KEY`
4. Railway will auto-detect Streamlit and deploy

## Features

### Phase 1 Pipeline

1. **Upload Phase**
   - Upload property photos
   - Set preferences (budget, style, goals)

2. **Valuation Agent**
   - Property assessment
   - Cost estimation
   - ROI analysis
   - Risk assessment

3. **Design Agent**
   - 3 design options generated
   - Room-by-room details
   - Color palettes

4. **Procurement Agent**
   - Bill of Materials
   - Supplier suggestions
   - Labor estimates

### Approval Gates

- Gate 1: After valuation (approve scope)
- Gate 2: After procurement (approve budget)

## Cost Estimation

Each full project run costs approximately:
- Valuation: ~$0.05-0.10 (Claude API)
- Design: ~$0.05-0.10 (Claude API)
- Procurement: ~$0.05-0.10 (Claude API)
- **Total per project: ~$0.15-0.30**

With $100/month budget: ~300-600 projects for testing

## Next Steps (Phase 2)

- [ ] Contractor Outreach Agent
- [ ] Scheduling Agent
- [ ] Monitoring Agent
- [ ] Audit Agent
- [ ] Audio AI Agent for calls

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Claude 3.5 Sonnet (Anthropic)
- **Deployment**: Streamlit Cloud / Railway

---

Built for OmniRenovation AI by the founding team.
