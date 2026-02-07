import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import time
import requests
import smtplib
import base64
from datetime import datetime, timedelta
from typing import TypedDict, List, Dict, Any, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import json
from io import BytesIO
from markdown_pdf import MarkdownPdf, Section
import tempfile

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from operator import add

from langchain_groq import ChatGroq
from typing_extensions import Annotated
from langchain_community.utilities import WikipediaAPIWrapper
from tavily import TavilyClient

import os
import streamlit as st

def get_secret(key: str, default=None):
    """
    Priority:
    1. Kubernetes / Docker environment variables
    2. Streamlit secrets.toml (local dev)
    """
    return os.getenv(key) or st.secrets.get(key, default)

# Set up page config
st.set_page_config(
    page_title="AI Trip Planner",
    layout="wide"
)

# Wikipedia Image Function (Enhanced)
def get_wikipedia_image(query: str, delay: float = 0.5) -> Optional[str]:
    """Fetch image URL from Tavily (replacing Wikipedia)"""
    time.sleep(delay)
    
    url = "https://api.tavily.com/search"
    headers = {
        "Authorization": f"Bearer {st.secrets.get('TAVILY_API_KEY', '')}",  
        "Content-Type": "application/json"
    }
    
    payload = {
        "query": query,
        "include_images": True,
        "limit": 1  # get top result
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if "images" in data and len(data["images"]) > 0:
            # Return the first image URL
            return data["images"][0].get("url")
    except Exception as e:
        print(f"Tavily image search error for '{query}': {e}")
    
    return None

@st.cache_resource
def initialize_apis():
    try:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=get_secret("GROQ_API_KEY"),
            temperature=0.7
        )

        tavily_client = TavilyClient(
            api_key=get_secret("TAVILY_API_KEY")
        )

        wikipedia = WikipediaAPIWrapper()
        return llm, tavily_client, wikipedia

    except Exception as e:
        st.error(f"Error initializing APIs: {e}")
        return None, None, None

# Enhanced state structure with Annotated fields for parallel updates
class TripPlannerState(TypedDict):
    origin: str 
    destination: str
    stopovers: str 
    start_date: str
    end_date: str
    budget: int
    num_travelers: int
    interests: List[str]
    # REMOVED: num_places and user_preferences
    
    # Agent outputs with Annotated for parallel updates
    destination_info: Dict[str, Any]
    destination_image: Optional[str]
    places_info: List[Dict[str, Any]]
    travel_options: List[Dict[str, Any]] 
    hotel_options: List[Dict[str, Any]]
    activities: List[Dict[str, Any]]
    weather_info: Dict[str, Any]
    final_itinerary: str
    
    # Progress tracking - using Annotated for parallel updates
    current_step: Annotated[List[str], add]
    error_messages: Annotated[List[str], add]
    progress: int

# Initialize APIs
llm, tavily_client, wikipedia = initialize_apis()

def research_agent(state: TripPlannerState) -> TripPlannerState:
    """Research destination AND stopovers, local costs, and get main image"""
    try:
        destination = state["destination"]
        stopovers = state.get("stopovers", "")
        
        st.write(f"🔍 Researching {destination} and stopovers ({stopovers})...")
        
        # Get Wikipedia information for both
        search_query = f"{destination} {stopovers} travel tourism"
        wiki_result = wikipedia.run(search_query)
        
        # Get destination image
        destination_image = get_wikipedia_image(f"{destination} landmark")
        
        # Extract key info using LLM
        prompt = f"""
        Extract key travel information from this Wikipedia content about {destination} and {stopovers}:
        {wiki_result[:2000]}
        
        Also consider general knowledge about these places.

        Return a JSON with:
        - description: 2-sentence overview of the main destination.
        - stopover_brief: Brief 1-sentence description of the stopovers ({stopovers}).
        - best_time_to_visit: season/months
        - currency: local currency
        - language: main language(s)
        - local_prices: Estimate cost for local things like boat rides, street food, taxi per km.
        - key_facts: 3 interesting facts as list
        """
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))
        
        try:
            destination_info = json.loads(content.replace("```json", "").replace("```", ""))
        except:
            destination_info = {
                "description": content[:300],
                "raw_info": wiki_result[:500]
            }
        
        return {
            "destination_info": destination_info,
            "destination_image": destination_image,
            "current_step": ["research_complete"],
            "progress": 20,
        }
    except Exception as e:
        return {
            "destination_info": {"error": str(e)},
            "destination_image": None,
            "error_messages": [f"Research error: {str(e)}"],
            "current_step": ["research_complete"],
        }


def places_agent(state: TripPlannerState) -> dict:
    """Agent to fetch top places using Tavily (for Destination AND Stopovers)"""
    try:
        destination = state["destination"]
        stopovers = state.get("stopovers", "")
        # REMOVED: limit from state, using default
        limit = 6 
        
        st.write(f"📍 Finding top places for {destination} and {stopovers}...")

        places_query = f"top tourist attractions in {destination} and {stopovers}"
        
        places_result = tavily_client.search(
            query=places_query,
            search_depth="basic",
            max_results=limit + 2 # Fetch a few more to cover both
        )

        places_info = []
        for r in places_result.get("results", [])[:limit]:
            place_details = {
                "name": r.get("title", ""),
                "rating": None,   
                "address": "",
                "types": [],
                "place_id": None,
                "image_url": None
            }
            # Optional: get image from Wikipedia
            place_image = get_wikipedia_image(f"{r.get('title', '')} {destination}")
            place_details["image_url"] = place_image

            places_info.append(place_details)

        return {
            "places_info": places_info,
            "current_step": ["places_complete"],
            "progress": 40
        }

    except Exception as e:
        return {
            "places_info": [],
            "error_messages": [f"Places error: {str(e)}"],
            "current_step": ["places_complete"],
        }


def weather_agent(state: TripPlannerState) -> dict:
    """Agent to fetch weather info"""
    try:
        destination = state["destination"]
        st.write(f"⛅ Checking weather for {destination}...")

        weather_query = f"current weather and climate in {destination}"
        weather_result = tavily_client.search(
            query=weather_query,
            search_depth="basic",
            max_results=3
        )

        weather_info = {
            "forecast": weather_result.get("results", [{}])[0].get("content", "Weather data unavailable"),
            "temperature": "Check local forecast",   
            "conditions": "See forecast details"
        }

        return {
            "weather_info": weather_info,
            "current_step": ["weather_complete"],
            "progress": 50
        }

    except Exception as e:
        return {
            "weather_info": {},
            "error_messages": [f"Weather error: {str(e)}"],
            "current_step": ["weather_complete"],
        }


def activities_agent(state: TripPlannerState) -> TripPlannerState:
    """Find activities and events using Tavily"""
    try:
        destination = state["destination"]
        interests = state["interests"]
        
        st.write(f"🎯 Finding activities in {destination}...")
        
        activities = []
        for interest in interests[:3]:
            query = f"{interest} activities events {destination} 2024 things to do cost price"
            results = tavily_client.search(
                query=query,
                search_depth="basic",
                max_results=3
            )
            
            for result in results.get("results", []):
                activity = {
                    "title": result.get("title", ""),
                    "description": result.get("content", "")[:200],
                    "category": interest,
                    "url": result.get("url", ""),
                    "image_url": get_wikipedia_image(f"{interest} {destination}")
                }
                activities.append(activity)
        
        return {
            "activities": activities[:8],
            "current_step": ["activities_complete"],
            "progress": 90,

        }
    except Exception as e:
        return {
            "activities": [],
            "error_messages": [f"Activities error: {str(e)}"],
            "current_step": ["activities_complete"],
        }

def itinerary_agent(state: TripPlannerState) -> TripPlannerState:
    """Generate final comprehensive itinerary"""
    try:
        st.write("📝 Generating your personalized itinerary...")
        
        start = datetime.strptime(state["start_date"], "%Y-%m-%d")
        end = datetime.strptime(state["end_date"], "%Y-%m-%d")
        duration = (end - start).days
        
        # REMOVED: Must Visit Places / User Preferences logic from prompt
        prompt = f"""
        Create a detailed {duration}-day trip itinerary starting from {state["origin"]}, visiting {state["stopovers"]}, and ending in {state["destination"]}.
        
        TRIP STRUCTURE:
        1. Start: {state["origin"]}
        2. Stopovers (En-route): {state["stopovers"]}
        3. Main Destination: {state["destination"]}
        
        TRIP DETAILS:
        - Dates: {state["start_date"]} to {state["end_date"]} ({duration} days)
        - Budget: ${state["budget"]} for {state["num_travelers"]} travelers
        - Interests: {", ".join(state["interests"])}
        
        AVAILABLE DATA:
        Destination/Stopover Info: {json.dumps(state.get("destination_info", {}))}
        Top Places Found: {json.dumps([p.get("name", "") for p in state.get("places_info", [])])}
        Travel Options: {json.dumps(state.get("travel_options", []))}
        Hotel Options: {json.dumps(state.get("hotel_options", []))}
        
        CREATE A CHRONOLOGICAL ITINERARY:
        
        ## 🗓️ TRIP OVERVIEW
        - Route: {state["origin"]} -> {state["stopovers"]} -> {state["destination"]}
        - Total Duration & Budget
        
        ## 🚆 MULTI-CITY TRAVEL PLAN
        - **Leg 1:** {state["origin"]} to {state["stopovers"]} (Suggest Train/Bus/Taxi options & duration)
        - **Leg 2:** {state["stopovers"]} to {state["destination"]}
        - **Return:** {state["destination"]} back to {state["origin"]}
        
        ## 🏨 ACCOMMODATION PLAN
        - Suggest where to stay (e.g., "Stay 1 night in [Stopover] and rest in [Destination]").
        
        ## 📅 DAY-BY-DAY ITINERARY
        - Clearly mark which city the user is in for each day.
        - **Stopover Days:** What to see in {state["stopovers"]}.
        - **Destination Days:** What to see in {state["destination"]}.
        - **LOCAL TRANSPORT:** How to move *between* cities and *within* cities.
        - **LOCAL COSTS:** Specific prices for inter-city travel (e.g., "Bus to Jaipur approx $10").
        
        ## 💰 BUDGET BREAKDOWN
        - Inter-city Travel: Estimated total
        - Hotels: Cost per city
        - Food & Activities: Daily estimate
        
        Make it practical, logical (don't zigzag), and well-formatted.
        """
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        return {
            "final_itinerary": content,
            "current_step": ["complete"],
            "progress": 100,
        }
    except Exception as e:
        return {
            "final_itinerary": f"Error generating itinerary: {str(e)}",
            "error_messages": [f"Itinerary error: {str(e)}"],
        }

def modify_itinerary(current_itinerary: str, instruction: str) -> str:
    """Modify the itinerary based on user instructions."""
    try:
        prompt = f"""
        You are an expert travel assistant. 
        Here is the current itinerary:
        {current_itinerary}

        User Instruction: "{instruction}"

        Please rewrite the itinerary to incorporate the user's changes. 
        Keep the format consistent with the original itinerary unless asked otherwise.
        Do not add any conversational filler, just return the updated itinerary.
        """
        response = llm.invoke(prompt)
        return getattr(response, "content", str(response))
    except Exception as e:
        return f"Error modifying itinerary: {str(e)}"


# PDF Generation Function
def generate_pdf(itinerary_text: str, destination: str, state) -> BytesIO:
    """Generate PDF from Markdown text."""
    pdf_buffer = BytesIO()
    pdf = MarkdownPdf(toc_level=0)
    title_md = f"# Trip Itinerary: {state.get('origin', 'Origin')} -> {state.get('stopovers', 'Stopover')} -> {destination}\n\n"
    full_md = title_md + itinerary_text
    pdf.add_section(Section(full_md))
    pdf.save(pdf_buffer)
    pdf_buffer.seek(0)
    return pdf_buffer
    
# Email function
def send_email(recipient_email: str, pdf_buffer: BytesIO, destination: str):
    """Send itinerary PDF via email using Gmail SMTP"""
    try:
        # Reset buffer position
        pdf_buffer.seek(0)

        sender_email = st.secrets.get("GMAIL_USER", "")
        sender_password = st.secrets.get("GMAIL_APP_PASSWORD", "")
        
        if not sender_email or not sender_password:
            st.error("Email credentials not configured in secrets!")
            return False
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f"Your Trip Itinerary for {destination}"
        
        body = f"""
        Hello!
        
        Your personalized trip itinerary for {destination} is ready!
        Please find the detailed PDF itinerary attached.
        
        Happy travels!
        
        Best regards,
        AI Trip Planner
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(pdf_buffer.read())
        encoders.encode_base64(part)
        part.add_header(
            'Content-Disposition',
            f'attachment; filename= "{destination.replace(" ", "_")}_itinerary.pdf"',
        )
        msg.attach(part)
        
        # Send email
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        return True
        
    except Exception as e:
        st.error(f"Email error: {str(e)}")
        return False

def travel_agent(state: TripPlannerState) -> TripPlannerState:
    """Search Flights, Trains, and Buses (RedBus)"""
    try:
        origin = state["origin"]
        destination = state["destination"]
        stopovers = state.get("stopovers", "")
        start_date = state["start_date"]
        
        travel_query = f"travel from {origin} to {destination} via {stopovers} flights train routes bus redbus price {start_date}"
        
        travel_results = tavily_client.search(
            query=travel_query,
            search_depth="advanced",
            max_results=6
        )
        
        prompt = f"""
        From these travel search results for traveling from {origin} to {destination} (potentially stopping at {stopovers}):
        {json.dumps([r.get('title', '') + ' - ' + r.get('content', '')[:150] for r in travel_results.get('results', [])])}
        
        Extract travel options as JSON array. 
        Focus on identifying connections: 
        1. {origin} -> {stopovers}
        2. {stopovers} -> {destination}
        
        Format:
        [
          {{
            "type": "Flight/Train/Bus",
            "route": "City A to City B",
            "provider": "Airline/Train/Bus Name",
            "price": "estimated price",
            "duration": "duration",
            "booking_info": "where to book"
          }}
        ]
        """
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        try:
            travel_options = json.loads(content.replace("```json", "").replace("```", ""))
            if not isinstance(travel_options, list):
                travel_options = []
        except:
            travel_options = []
        
        return {
            "travel_options": travel_options,
            "current_step": ["travel_complete"], 
        }
    except Exception as e:
        return {
            "travel_options": [],
            "error_messages": [f"Travel search error: {str(e)}"],
            "current_step": ["travel_complete"],
        }

def hotels_agent(state: TripPlannerState) -> TripPlannerState:
    """Search hotels using Tavily"""
    try:
        destination = state["destination"]
        stopovers = state.get("stopovers", "")
        budget = state["budget"]
        
        hotel_query = f"best hotels in {destination} and {stopovers} accommodation prices {budget} budget"
        
        hotel_results = tavily_client.search(
            query=hotel_query,
            search_depth="advanced",
            max_results=6
        )
        
        prompt = f"""
        From these hotel search results for {destination} and {stopovers}:
        {json.dumps([r.get('title', '') + ' - ' + r.get('content', '')[:150] for r in hotel_results.get('results', [])])}
        
        Extract 3-5 hotel options as JSON array (Include options for both cities if possible):
        [
          {{
            "name": "hotel name",
            "city": "City Name",
            "price_per_night": "price per night",
            "rating": "star rating",
            "booking_platform": "where to book"
          }}
        ]
        """
        
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response))

        try:
            hotel_options = json.loads(content.replace("```json", "").replace("```", ""))
            if not isinstance(hotel_options, list):
                hotel_options = []
        except:
            hotel_options = []
        
        return {
            "hotel_options": hotel_options,
            "current_step": ["hotels_complete"],
        }
        
    except Exception as e:
        return {
            "hotel_options": [],
            "error_messages": [f"Hotel search error: {str(e)}"],
            "current_step": ["hotels_complete"],
        }

# Create enhanced workflow with proper parallel execution
def create_workflow():
    workflow = StateGraph(TripPlannerState)
    
    # Add nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("places", places_agent)
    workflow.add_node("weather", weather_agent)
    workflow.add_node("travel", travel_agent) 
    workflow.add_node("hotels", hotels_agent)
    workflow.add_node("activities", activities_agent)
    workflow.add_node("itinerary", itinerary_agent)

     # Gate node to wait for all parallel branches
    workflow.add_node("sync_gate", lambda s: s) 

    # Define the flow
    workflow.set_entry_point("research")
    
    # Sequential flow: research -> places
    workflow.add_edge("research", "places")
    
    # Parallel execution: places -> [travel, hotels, weather]
    workflow.add_edge("places", "travel")
    workflow.add_edge("places", "hotels")
    workflow.add_edge("places", "weather")

    # Wait for all parallel agents before activities
    workflow.add_edge("travel", "sync_gate")
    workflow.add_edge("hotels", "sync_gate")
    workflow.add_edge("weather", "sync_gate")

    # Barrier condition: proceed only when ALL parallel branches are done
    def _barrier_condition(state: TripPlannerState) -> str:
        steps = set(state.get("current_step", []))
        required = {"travel_complete", "hotels_complete", "weather_complete"}
        return "go" if required.issubset(steps) else "wait"

    workflow.add_conditional_edges(
        "sync_gate",
        _barrier_condition,
        {
            "go": "activities",   
            "wait": END,   
        },
    )
  
    # Final steps
    workflow.add_edge("activities", "itinerary")
    workflow.add_edge("itinerary", END)
    
    return workflow.compile()

# Enhanced Streamlit UI
def main():
    st.title("AI Multi-Agent Trip Planner")
    st.markdown("Plan your trip with AI-powered research, real-time search, and beautiful itineraries!")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        if llm and tavily_client:
            st.success("All APIs connected!")
        else:
            st.error("Check API keys in secrets")
            st.code("""
            # Add to .streamlit/secrets.toml:
            GROQ_API_KEY = "your_key"
            TAVILY_API_KEY = "your_key"
            GMAIL_USER = "your_gmail@gmail.com"
            GMAIL_APP_PASSWORD = "your_app_password"
            """)
        
        st.markdown("---")
        st.markdown("### 🔄 Workflow")
        st.markdown("""
        1. **Research** destination & local costs
        2. **Places & Weather** (parallel)
        3. **Travel (Flight/Train/Bus) & Hotels** (parallel)
        4. **Activities** based on interests
        5. **Generate** final itinerary
        """)
        st.markdown("---")
        if st.button("🗑️ Clear Trip Cache", help="Clear cached results to force new planning"):
            # Clear trip-related session state
            keys_to_clear = [k for k in st.session_state.keys() if k.startswith(('trip_', 'email_sent_', 'pdf_buffer'))]
            for key in keys_to_clear:
                del st.session_state[key]
            st.rerun()

    
    # Main form
    with st.form("enhanced_trip_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            origin = st.text_input("🏠 Starting From (Origin)", placeholder="e.g., New Delhi")
            destination = st.text_input("🏙️ Main Destination", placeholder="e.g., Jaipur, Rajasthan")
            stopovers = st.text_input("🛑 Stopovers / En-route Cities (Optional)", placeholder="e.g., Khatushyam Ji, Agra")
            
            start_date = st.date_input("📅 Start Date", datetime.now() + timedelta(days=7))
            budget = st.number_input("💰 Budget (USD)", min_value=0, max_value=50000, value=1000, step=50)
        
        with col2:
            end_date = st.date_input("📅 End Date", datetime.now() + timedelta(days=14))
            num_travelers = st.number_input("👥 Travelers", min_value=1, max_value=10, value=2)
            interests = st.multiselect(
                "🎯 Your Interests",
                ["Culture", "Food", "Adventure", "Museums", "Nature", "Shopping", "Nightlife", "History", "Art", "Music", "Architecture", "Local Experiences"],
                default=["Culture", "Food", "Local Experiences"]
            )
        
        # REMOVED: Trip Preferences (Slider & Text Area)

        submitted = st.form_submit_button("🚀 Plan My Amazing Trip!", use_container_width=True)
    
    trip_key = None
    if destination and origin:
        # Removed num_places from key
        trip_key = f"trip_{origin}_{destination}_{stopovers}_{start_date}_{end_date}_{budget}"

    if submitted and destination and origin and llm:
        if start_date >= end_date:
            st.error("End date must be after start date!")
            return
        
        if not interests:
            st.error("Please select at least one interest!")
            return
        
        if trip_key and trip_key in st.session_state:
            final_state = st.session_state[trip_key]
            st.info("📋 Using cached trip plan (change inputs to regenerate)")
        else:
            with st.spinner("Planning your trip..."):
                initial_state = TripPlannerState(
                    origin=origin, 
                    destination=destination,
                    stopovers=stopovers, 
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    budget=budget,
                    num_travelers=num_travelers,
                    interests=interests,
                    # REMOVED: num_places and user_preferences from initial state
                    destination_info={},
                    destination_image=None,
                    places_info=[],
                    travel_options=[], 
                    hotel_options=[],
                    activities=[],
                    weather_info={},
                    final_itinerary="",
                    current_step=["starting"],
                    error_messages=[],
                    progress=0
                )
                
                progress_bar = st.progress(0)
                status_container = st.container()
                
                try:
                    app = create_workflow()
                    final_state = app.invoke(initial_state)
                    
                    if trip_key:
                        st.session_state[trip_key] = final_state
                    
                    progress_bar.progress(100)
                
                except Exception as e:
                    st.error(f"❌ Workflow execution error: {str(e)}")
                    st.info("Please check your API keys and internet connection.")
                    return

    if trip_key and trip_key in st.session_state:
        final_state = st.session_state[trip_key]
        
        if final_state.get("final_itinerary"):
            if final_state.get("destination_image"):
                st.image(final_state["destination_image"], caption=f"{destination}", use_column_width=True)
            
            st.markdown("## Your Complete Trip Itinerary")
            st.markdown(final_state["final_itinerary"])
        
            col1, col2, col3 = st.columns(3)
            
            with col1:
                with st.expander("📝 Destination & Stopover Details"):
                    dest_info = final_state.get("destination_info", {})
                    if dest_info:
                        st.json(dest_info)
            
            with col2:
                with st.expander("📍 Top Places & Attractions"):
                    places = final_state.get("places_info", [])
                    for i, place in enumerate(places[:5], 1):
                        st.write(f"**{i}. {place.get('name', 'Unknown')}**")
                        if place.get('image_url'):
                            st.image(place['image_url'], width=200)
                        st.write("---")
            
            with col3:
                with st.expander("🚆 Travel Options (Train/Bus/Flight)"):
                    travel_opts = final_state.get("travel_options", [])
                    if travel_opts:
                        for i, opt in enumerate(travel_opts, 1):
                            st.write(f"**Option {i}: {opt.get('type', 'Transport')}**")
                            st.write(f"🛣️ Route: {opt.get('route', 'N/A')}")
                            st.write(f"🏢 Provider: {opt.get('provider', 'N/A')}")
                            st.write(f"💰 Price: {opt.get('price', 'N/A')}")
                            st.write(f"ℹ️ Booking: {opt.get('booking_info', 'N/A')}")
                            st.write("---")
                    else:
                        st.info("No specific travel details found. Check RedBus or IRCTC manually.")
            
            st.markdown("---")
            st.subheader("💬 Chat with AI to Modify Itinerary")
            
            # Initialize chat history
            if f"chat_history_{trip_key}" not in st.session_state:
                st.session_state[f"chat_history_{trip_key}"] = []

            # Display chat messages from history on app rerun
            for message in st.session_state[f"chat_history_{trip_key}"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # React to user input
            if prompt := st.chat_input("Make changes (e.g., 'Add a dinner at Chokhi Dhani on Day 2')"):
                # Display user message in chat message container
                st.chat_message("user").markdown(prompt)
                # Add user message to chat history
                st.session_state[f"chat_history_{trip_key}"].append({"role": "user", "content": prompt})
                
                with st.spinner("Updating itinerary..."):
                    # Modify the itinerary
                    updated_itinerary = modify_itinerary(final_state["final_itinerary"], prompt)
                    
                    # Update state
                    final_state["final_itinerary"] = updated_itinerary
                    st.session_state[trip_key] = final_state
                    
                    # Display assistant response in chat message container
                    with st.chat_message("assistant"):
                        st.markdown("I've updated your itinerary based on your request. Please scroll up to see the changes.")
                    
                    # Add assistant response to chat history
                    st.session_state[f"chat_history_{trip_key}"].append({"role": "assistant", "content": "I've updated your itinerary based on your request. Please scroll up to see the changes."})
                    
                    # Force rerun to update the main itinerary view
                    st.rerun()

            
            # PDF Generation
            if f"pdf_buffer_{trip_key}" not in st.session_state:
                st.session_state[f"pdf_buffer_{trip_key}"] = generate_pdf(
                    final_state["final_itinerary"], 
                    destination, 
                    final_state
                )
            
            pdf_buffer = st.session_state[f"pdf_buffer_{trip_key}"]
            
            st.markdown("### 📥 Download or Email Itinerary")
            
            col_download, col_email = st.columns(2)
            
            with col_download:
                st.download_button(
                    label="📄 Download PDF",
                    data=pdf_buffer.getvalue(),
                    file_name=f"{destination.replace(' ', '_')}_itinerary.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            
            with col_email:
                with st.form(key=f"email_form_{trip_key}"):
                    st.write("📧 **Send to Email**")
                    email_input = st.text_input("Enter Email Address")
                    send_btn = st.form_submit_button("Send Itinerary")
                    
                    if send_btn and email_input:
                        with st.spinner("Sending email..."):
                            if send_email(email_input, pdf_buffer, destination):
                                st.success(f"✅ Sent to {email_input}!")
                            else:
                                st.error("❌ Failed. Check secrets.toml settings.")

    elif submitted and not llm:
        st.error("❌ Please configure your API keys first!")
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    🤖 Powered by LangGraph Multi-Agent System | Built with Streamlit<br>
    Uses: Gemini LLM • Tavily Search • Google Places • Wikipedia
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()