# AI Trip Planner

The **Multi-Agent Trip Planner** is an intelligent travel planning application built using **LangGraph**, **Google Gemini API**, **Tavily API**, and **Wikipedia**.

It leverages a coordinated set of specialized AI agents to research destinations, analyze travel options (including Flights, Trains, and Buses), and generate a personalized, end-to-end travel itinerary. The system produces detailed, user-specific itineraries with real-time insights, local cost estimates, and supports exporting the final plan as a **PDF** or sending it directly via **Email**.

---

##  Key Features

-  **Multi-Agent Architecture** orchestrated with LangGraph
-  **Origin-Based Planning**: customizable starting point for accurate travel routing.
-  **Multi-Mode Transport Search**: Compares **Flights**, **Trains**, and **Buses** to find the best route.
-  **Destination Research** using Wikipedia and Tavily Search.
-  **Smart Budgeting & Local Costs**: Handles low-budget trips and estimates local costs (e.g., taxis, street food).
-  **Weather Forecast Integration**.
-  **Hotel Suggestions** tailored to your budget.
-  **Activity Planning** based on user interests.
-  **Email Integration**: Send the generated PDF itinerary directly to your inbox.
-  **AI-Generated Day-by-Day Itinerary**.
-  **PDF Export** for offline access.

---

### Application UI

![Screen 1](./demo/image1.png)
![Screen 2](./demo/image2.png)
![Screen 3](./demo/image3.png)
![Screen 4](./demo/image4.png)
![Screen 5](./demo/image5.png)
![Screen 6](./demo/image6.png)
![Screen 7](./demo/image7.png)
![Screen 8](./demo/image8.png)
![Screen 9](./demo/image9.png)
![Screen 10](./demo/image10.png)
![Screen 11](./demo/image11.png)
![Screen 12](./demo/image12.png)
![Sample PDF](./demo/image13.png)

---

## 📦 Installation

### 1. Clone the Repository
```bash
git clone [https://github.com/sakshivedi-1/AI_TRIP_PLANNER.git](https://github.com/sakshivedi-1/AI_TRIP_PLANNER.git)
cd ai-trip-planner

python -m venv .venv
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows

pip install -r requirements.txt

### 4. API Key & Email Configuration
This application requires credentials for Google Gemini, Tavily, and Gmail (for the email feature).

Create a .streamlit/secrets.toml file (Recommended for Streamlit):

GOOGLE_API_KEY = "your_google_gemini_api_key"
TAVILY_API_KEY = "your_tavily_api_key"

# Optional: Required only if you want to use the Email feature
GMAIL_USER = "your_email@gmail.com"
GMAIL_APP_PASSWORD = "your_app_password"

Note: For GMAIL_APP_PASSWORD, go to your Google Account > Security > 2-Step Verification > App Passwords to generate a 16-digit password. DO NOT use your regular Gmail password.

### Running the Application
Bash
```
streamlit run app.py
```

The application will be available at: http://localhost:8501

### 📚 Technology Stack
LangGraph — Multi-agent orchestration

LangChain — LLM pipeline integration

Google Gemini API — Reasoning, itinerary generation, and cost estimation

Tavily API — Real-time web search (Flights, Trains, RedBus, Hotels)

Wikipedia API — Destination knowledge enrichment

Streamlit — Interactive frontend

ReportLab / Markdown-PDF — PDF generation

SMTP (Gmail) — Email delivery of itineraries

### 🔄 System Workflow
Code snippet

flowchart TD
    A[Research Agent] --> B[Places Agent]
    B --> C[Travel Agent]
    B --> D[Hotels Agent]
    B --> E[Weather Agent]

    C[Travel Agent (Flight/Train/Bus)] --> F[Sync Gate]
    D[Hotels Agent] --> F[Sync Gate]
    E[Weather Agent] --> F[Sync Gate]

    F -->|All complete| G[Activities Agent]
    G --> H[Itinerary Agent]
    H --> I[Final Output: PDF / Email]
Workflow Explanation
Research Agent: Gathers foundational destination info and estimates local living costs.

Places Agent: Identifies top tourist attractions.

#### Parallel Agents:

Travel Agent: Searches for Flights, Trains, and Buses (e.g., RedBus) from the Origin.

Hotels Agent: Finds accommodation fitting the budget.

Weather Agent: Checks the forecast.

Sync Gate: Ensures all parallel research is finished before proceeding.

Activities Agent: Curates personalized experiences based on user interests.

Itinerary Agent: Assembles all data into a cohesive day-by-day plan.

User Output: Delivered as a displayed itinerary, downloadable PDF, and via Email.