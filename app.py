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

from langchain_google_genai import ChatGoogleGenerativeAI
from typing_extensions import Annotated
from langchain_community.utilities import WikipediaAPIWrapper
from tavily import TavilyClient

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

# Initialize APIs
@st.cache_resource
def initialize_apis():
    """Initialize all API clients"""
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=st.secrets.get("GOOGLE_API_KEY", ""),
            temperature=0.7
        )
        
        tavily_client = TavilyClient(api_key=st.secrets.get("TAVILY_API_KEY", ""))
        wikipedia = WikipediaAPIWrapper()
        
        return llm, tavily_client, wikipedia
    except Exception as e:
        st.error(f"Error initializing APIs: {str(e)}")
        return None, None, None

# Enhanced state structure with Annotated fields for parallel updates
class TripPlannerState(TypedDict):
    origin: str  # Added Origin
    destination: str
    start_date: str
    end_date: str
    budget: int
    num_travelers: int
    interests: List[str]
    
    # Agent outputs with Annotated for parallel updates
    destination_info: Dict[str, Any]
    destination_image: Optional[str]
    places_info: List[Dict[str, Any]]
    # Renamed to general travel options to include trains/buses
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
    """Research destination, local costs, and get main image"""
    try:
        destination = state["destination"]
        st.write(f"🔍 Researching {destination} and local costs...")
        
        # Get Wikipedia information
        wiki_result = wikipedia.run(f"{destination} travel tourism")
        
        # Get destination image
        destination_image = get_wikipedia_image(f"{destination} landmark")
        
        # Extract key info using LLM - ADDED request for local prices
        prompt = f"""
        Extract key travel information from this Wikipedia content about {destination}:
        {wiki_result[:2000]}
        
        Also consider general knowledge about {destination} to answer the following.

        Return a JSON with:
        - description: 2-sentence overview
        - best_time_to_visit: season/months
        - currency: local currency
        - language: main language(s)
        - timezone: timezone info
        - local_prices: Estimate cost for local things like boat rides, street food, taxi per km (be specific about low-cost options).
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
        }


def places_agent(state: TripPlannerState) -> dict:
    """Agent to fetch top places using Tavily"""
    try:
        destination = state["destination"]
        st.write(f"📍 Finding places for {destination}...")

        places_query = f"top tourist attractions in {destination}"
        places_result = tavily_client.search(
            query=places_query,
            search_depth="basic",
            max_results=6
        )

        places_info = []
        for r in places_result.get("results", [])[:6]:
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
            "error_messages": [f"Places error: {str(e)}"]
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
            "error_messages": [f"Weather error: {str(e)}"]
        }


def activities_agent(state: TripPlannerState) -> TripPlannerState:
    """Find activities and events using Tavily"""
    try:
        destination = state["destination"]
        interests = state["interests"]
        
        st.write(f"🎯 Finding activities in {destination}...")
        
        activities = []
        for interest in interests[:3]:
            # Modified query to find local prices for activities too
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
        }

def itinerary_agent(state: TripPlannerState) -> TripPlannerState:
    """Generate final comprehensive itinerary"""
    try:
        st.write("📝 Generating your personalized itinerary...")
        
        # Calculate trip duration
        start = datetime.strptime(state["start_date"], "%Y-%m-%d")
        end = datetime.strptime(state["end_date"], "%Y-%m-%d")
        duration = (end - start).days
        
        # ADDED: Explicit instructions for Origin, Local Prices, and Transport
        prompt = f"""
        Create a detailed {duration}-day trip itinerary for {state["destination"]} starting from {state["origin"]}.
        
        TRIP DETAILS:
        - Origin: {state["origin"]}
        - Destination: {state["destination"]}
        - Dates: {state["start_date"]} to {state["end_date"]} ({duration} days)
        - Budget: ${state["budget"]} for {state["num_travelers"]} travelers
        - Interests: {", ".join(state["interests"])}
        
        AVAILABLE DATA:
        Destination Info: {json.dumps(state.get("destination_info", {}))}
        Top Places: {json.dumps([p.get("name", "") for p in state.get("places_info", [])[:5]])}
        Travel Options (Flights/Trains/Bus): {json.dumps(state.get("travel_options", []))}
        Hotel Options: {json.dumps(state.get("hotel_options", []))}
        Activities: {json.dumps([a.get("title", "") for a in state.get("activities", [])[:6]])}
        Weather: {json.dumps(state.get("weather_info", {}))}
        
        CREATE A COMPREHENSIVE ITINERARY WITH:
        
        ## 🗓️ TRIP OVERVIEW
        - Destination summary
        - Duration and dates
        - Total Estimated Cost breakdown
        
        ## 🚆 TRAVEL & ARRIVAL (IMPORTANT)
        - Compare Flights vs Trains vs Bus (RedBus) based on the provided Travel Options.
        - Recommend the best mode of transport from {state["origin"]} to {state["destination"]} considering the budget of ${state["budget"]}.
        
        ## 🏨 ACCOMMODATION
        - Top recommendations from search results.
        
        ## 📅 DAY-BY-DAY ITINERARY
        For each day include:
        - Morning/Afternoon/Evening activities.
        - **LOCAL TRANSPORT:** Suggest how to get around (e.g., "Take a boat ride for approx $2", "Auto-rickshaw cost approx $5").
        - **LOCAL COSTS:** Mention prices for food/entry tickets suitable for a ${state["budget"]} budget.
        
        ## 💰 BUDGET BREAKDOWN
        - Travel (From {state["origin"]}): estimated cost
        - Hotels: per night cost
        - Activities & Local Transport: daily budget (Include specific local prices like boat rides if applicable)
        - Food: daily budget
        
        ## 📋 PRACTICAL TIPS
        - Safety & Scams
        - Local apps (RedBus, Uber/Ola equivalents)
        
        Make it engaging, practical, and well-formatted.
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

# PDF Generation Function
def generate_pdf(itinerary_text: str, destination: str, state) -> BytesIO:
    """Generate PDF from Markdown text."""
    pdf_buffer = BytesIO()
    pdf = MarkdownPdf(toc_level=0)
    title_md = f"# Trip Itinerary: {destination} (From {state.get('origin', 'N/A')})\n\n"
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
        start_date = state["start_date"]
        
        # Updated Query to include Trains and RedBus
        travel_query = f"travel from {origin} to {destination} flights train routes bus redbus price booking {start_date}"
        
        travel_results = tavily_client.search(
            query=travel_query,
            search_depth="advanced",
            max_results=6
        )
        
        # Extract structured travel data
        prompt = f"""
        From these travel search results for traveling from {origin} to {destination}:
        {json.dumps([r.get('title', '') + ' - ' + r.get('content', '')[:150] for r in travel_results.get('results', [])])}
        
        Extract travel options as JSON array. Include Flights, Trains, and Buses (mention RedBus if found).
        Format:
        [
          {{
            "type": "Flight/Train/Bus",
            "provider": "Airline/Train Name/Bus Operator",
            "price": "estimated price",
            "duration": "duration",
            "booking_info": "where to book (e.g., RedBus, IRCTC)"
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
            "current_step": ["travel_complete"], # updated step name
        }
    except Exception as e:
        return {
            "travel_options": [],
            "error_messages": [f"Travel search error: {str(e)}"]
        }

def hotels_agent(state: TripPlannerState) -> TripPlannerState:
    """Search hotels using Tavily"""
    try:
        destination = state["destination"]
        budget = state["budget"]
        
        hotel_query = f"best hotels {destination} accommodation booking prices reviews {budget} budget"
        hotel_results = tavily_client.search(
            query=hotel_query,
            search_depth="advanced",
            max_results=6
        )
        
        prompt = f"""
        From these hotel search results for {destination}:
        {json.dumps([r.get('title', '') + ' - ' + r.get('content', '')[:150] for r in hotel_results.get('results', [])])}
        
        Extract 3-5 hotel options as JSON array:
        [
          {{
            "name": "hotel name",
            "price_per_night": "price per night",
            "rating": "star rating or review score",
            "location": "area/neighborhood",
            "amenities": "key amenities",
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
            "error_messages": [f"Hotel search error: {str(e)}"]
        }

# Create enhanced workflow with proper parallel execution
def create_workflow():
    workflow = StateGraph(TripPlannerState)
    
    # Add nodes
    workflow.add_node("research", research_agent)
    workflow.add_node("places", places_agent)
    workflow.add_node("weather", weather_agent)
    workflow.add_node("travel", travel_agent) # Renamed from flights
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
        # Updated check for travel_complete instead of flights_complete
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
            GOOGLE_API_KEY = "your_key"
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
            # ADDED: Origin Input
            origin = st.text_input("🏠 Starting From (Origin)", placeholder="e.g., New Delhi, India")
            destination = st.text_input("🏙️ Destination", placeholder="e.g., Goa, India")
            start_date = st.date_input("📅 Start Date", datetime.now() + timedelta(days=7))
            # CHANGED: Min budget to 0 to allow low starting price
            budget = st.number_input("💰 Budget (USD)", min_value=0, max_value=50000, value=1000, step=50)
        
        with col2:
            end_date = st.date_input("📅 End Date", datetime.now() + timedelta(days=14))
            num_travelers = st.number_input("👥 Travelers", min_value=1, max_value=10, value=2)
            interests = st.multiselect(
                "🎯 Your Interests",
                ["Culture", "Food", "Adventure", "Museums", "Nature", "Shopping", "Nightlife", "History", "Art", "Music", "Architecture", "Local Experiences"],
                default=["Culture", "Food", "Local Experiences"]
            )
        
        submitted = st.form_submit_button("🚀 Plan My Amazing Trip!", use_container_width=True)
    
    trip_key = None
    if destination and origin:
        trip_key = f"trip_{origin}_{destination}_{start_date}_{end_date}_{budget}_{num_travelers}"

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
                    origin=origin, # Include Origin in state
                    destination=destination,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    budget=budget,
                    num_travelers=num_travelers,
                    interests=interests,
                    destination_info={},
                    destination_image=None,
                    places_info=[],
                    travel_options=[], # Changed from flight_options
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
                with st.expander("📝 Destination Details"):
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
                # CHANGED: Display Travel Options (Trains/Flights/Bus)
                with st.expander("🚆 Travel Options (Train/Bus/Flight)"):
                    travel_opts = final_state.get("travel_options", [])
                    if travel_opts:
                        for i, opt in enumerate(travel_opts, 1):
                            st.write(f"**Option {i}: {opt.get('type', 'Transport')}**")
                            st.write(f"🏢 Provider: {opt.get('provider', 'N/A')}")
                            st.write(f"💰 Price: {opt.get('price', 'N/A')}")
                            st.write(f"⏱️ Duration: {opt.get('duration', 'N/A')}")
                            st.write(f"ℹ️ Booking: {opt.get('booking_info', 'N/A')}")
                            st.write("---")
                    else:
                        st.info("No specific travel details found. Check RedBus or IRCTC manually.")
            
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
            
            # ADDED: Functional Email Section
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