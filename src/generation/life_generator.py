# src/generation/life_generator.py
# Contains the functions from the provided background generation script.
# Updated imports to use the correct models file.
# Added Rich formatting for console output during generation.

import re
import math
import logging
import asyncio
import json
import calendar
import os
import re
from datetime import date, timedelta, datetime, timezone
from typing import Dict, Any, Optional, List, Tuple, Type, TypeVar, Union # Ensure Optional and Tuple are imported
import time
from pydantic import BaseModel, ValidationError
from duckduckgo_search import DDGS

# Use models from this directory
from src.generation.llm_service import LLMService
from src.generation.models import (
    InitialRelationshipsResponse, Person, YearlySummariesResponse,
    MonthlySummariesResponse, DailySummariesResponse, HourlyBreakdownResponse,
    YearSummary, MonthSummary, DaySummary, HourActivity, PersonaDetailsResponse
)

# Import Rich
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.pretty import pretty_repr
from rich.tree import Tree
# Standard logging for background info
logger = logging.getLogger(__name__)
# Rich console for user-facing status updates during generation
console = Console()
# --- End Logger Config ---

# --- Constants ---
# Define the directory relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIFE_SUMMARY_DIR = os.path.join(PROJECT_ROOT, "data", "life_summaries")

T = TypeVar('T', bound='BaseModel')

# --- Utility function to extract location (No changes) ---
def extract_location_from_text(text: str, default: str = "Unknown Location") -> str:
    patterns = [
        r'(?:lived|resided|based|worked|studied|grew up)\s+in\s+([\w\s,-]+)',
        r'moved\s+to\s+([\w\s,-]+)',
        r'born\s+in\s+([\w\s,-]+)',
        r'location\s*[:=]\s*([\w\s,-]+)',
        r'settled\s+in\s+([\w\s,-]+)'
    ]
    found_locations = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            location = match.strip().split(', where')[0].split(', which')[0].split(' during')[0].split(' in the year')[0].strip()
            location = re.sub(r'^[.,;\s]+|[.,;\s]+$', '', location)
            if location and len(location) > 1:
                if location.lower() not in ["the city", "a town", "the country", "home", "school", "university", "college"]:
                    found_locations.append(location)
    if found_locations:
        primary_location = found_locations[-1]
        logger.debug(f"Extracted location '{primary_location}' from text.")
        return primary_location
    else:
        logger.warning(f"Could not extract specific location from text snippet: '{text[:100]}...' Using default.")
        return default

# --- Helper Function to call LLM and validate (No changes) ---
async def _call_llm_and_get_validated_data(
    llm_service: LLMService,
    prompt: str,
    response_model: Type[T],
    operation_description: str
) -> Optional[Dict]:
    logger.debug(f"Calling LLM for '{operation_description}'...")
    try:
        response_dict = await llm_service.generate_content(
            prompt=prompt,
            response_model=response_model
        )
        if not response_dict:
            logger.error(f"LLM call for '{operation_description}' returned None or empty.")
            return None
        if "error" in response_dict:
            error_msg = response_dict["error"]
            logger.error(f"LLM service returned error during '{operation_description}': {error_msg}")
            if "raw_response" in response_dict:
                 logger.error(f"Raw response was: {response_dict['raw_response']}")
            return None
        if "error" in response_dict: # Double check
             logger.error(f"Validation likely failed for '{operation_description}'. Error: {response_dict['error']}")
             return None
        logger.debug(f"Successfully received validated data for '{operation_description}'.")
        return response_dict
    except Exception as e:
        logger.error(f"Exception calling LLM or processing result for '{operation_description}': {e}", exc_info=True)
        return None

async def generate_random_persona(
    llm_service: LLMService,
    world_type: str,
    world_description: str
) -> Optional[Dict[str, Any]]:
    """Generates a random plausible persona including age using the LLM, considering the world context."""
    logger.info(f"Generating random persona for world type '{world_type}'...")

    prompt = f"""
Create a detailed and plausible random fictional persona profile suitable for the following world:
World Type: {world_type}
World Description: {world_description}

**Instructions:**
- Generate a persona that fits logically within this world setting.
- Ensure the persona's details (occupation, background, location, etc.) are consistent with the provided world description.
- Age should be an integer between 18 and 45.

**Required Fields:** Name, Age (integer), Occupation, Current_location (City, State/Country appropriate for the world), Personality_Traits (list, 3-6 adjectives), Birthplace (City, State/Country appropriate for the world), Education, Physical_Appearance (brief description).

Respond ONLY with valid JSON matching the required fields.
"""

    # Assuming PersonaDetailsResponse is the correct model for validation here
    validated_data = await _call_llm_and_get_validated_data(llm_service, prompt, PersonaDetailsResponse, "random persona generation")
    if validated_data:
        logger.info(f"Successfully generated random persona: {validated_data.get('Name', 'Unknown')}, Age: {validated_data.get('Age', 'N/A')}")
        # console.print(Panel(pretty_repr(validated_data), title="Validated Persona Data", border_style="green", expand=True))
        return validated_data
    else:
        logger.error("Failed to generate or validate random persona.")
        return None

async def generate_initial_relationships( llm_service: LLMService, persona_details_str: str ) -> Optional[Dict[str, List[Dict]]]:
    """Generates the initial family structure (parents, siblings)."""
    logger.info("Generating initial relationship structure...")
    prompt = f"""
Based on the following persona details: {persona_details_str}
Establish a plausible immediate family structure: parents and any siblings. Include brief details.
Respond ONLY with JSON: {{"parents": [{{"name": "...", "relationship": "...", "details": "..."}}], "siblings": [...]}}

"""
    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, InitialRelationshipsResponse, "initial relationships" )
    if validated_data:
        logger.info(f"Generated initial relationships: {len(validated_data.get('parents',[]))} parents, {len(validated_data.get('siblings',[]))} siblings.")
        return validated_data
    else:
        logger.error("Failed to get validated initial relationships.");
        return None

async def generate_yearly_summaries(
    llm_service: LLMService, persona_details_str: str, initial_relationships_str: str,
    birth_year: int, last_year_to_generate: int, current_year: int, ddgs_instance: DDGS
) -> Optional[Tuple[Dict[int, Tuple[str, Optional[str], Optional[str]]], int, int]]:
    """Generates yearly summaries with location, birthday, and news searches."""
    logger.info(f"Generating yearly summaries with LOCATION JSON ({birth_year}-{last_year_to_generate})...")
    summaries_dict: Dict[int, Tuple[str, Optional[str], Optional[str]]] = {}
    processed_years = set()
    birth_month, birth_day = None, None

    news_context_by_year = {}
    for year in range(birth_year, last_year_to_generate + 1):
        # Perform a news search for the year
        time.sleep(1)
        search_query = f"major world events {year}"
        logger.info(f"Performing DDGS search for year {year}: '{search_query}'")
        search_results_summary = "Search placeholder."
        try:
            search_results_list: List[Dict] = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=5)
            if search_results_list:
                formatted_results = [f"- {r.get('title', '')}: {r.get('body', '')[:100]}..." for r in search_results_list]
                search_results_summary = "\n".join(formatted_results)
                console.print(Panel(
                    "\n".join(formatted_results),
                    title=f"News Search Results for {year}",
                    border_style="blue",
                    expand=True
                ))
            else:
                search_results_summary = "Search returned no results."
                console.print(Panel(
                    "No results found.",
                    title=f"News Search Results for {year}",
                    border_style="red",
                    expand=True
                ))
            logger.debug(f"DDGS yearly context summary for {year}:\n{search_results_summary}")
        except Exception as search_err:
            logger.error(f"DDGS search failed for year {year}: {search_err}")
            search_results_summary = "Use your internal knowledge of this years world and local events."
            console.print(Panel(
                "Search failed due to an error.",
                title=f"News Search Results for {year}",
                border_style="red",
                expand=True
            ))

        # Store the news context for the year
        news_context_by_year[year] = search_results_summary

    # Generate the yearly summary
    news_context_combined = "\n".join([f"{year}: {news}" for year, news in news_context_by_year.items()])
    prompt = f"""
    Persona details: {persona_details_str}
    Initial family: {initial_relationships_str}
    Born in {birth_year}. Summaries needed up to end of {last_year_to_generate}.
    News Context for all years:
    {news_context_combined}
    **Instructions:** Provide a summary of major events for the year, including the persona's location and key life events.
    **Output Format:** Respond ONLY with JSON: {{"birth_month": int, "birth_day": int, "summaries": [{{"year": int, "location": str, "summary": str}}]}}

    """
    validated_data = await _call_llm_and_get_validated_data(
        llm_service, prompt, YearlySummariesResponse, f"yearly summary for {birth_year}-{last_year_to_generate}"
    )
    if validated_data:
        try:
            # Extract birth month and day if available
            if year == birth_year:
                birth_month = validated_data.get("birth_month")
                birth_day = validated_data.get("birth_day")
                logger.info(f"Established Birthday: {birth_year}-{birth_month:02d}-{birth_day:02d}")

            # Process each summary in the "summaries" list
            for summary_item in validated_data.get("summaries", []):
                year_num = summary_item["year"]
                summary_text = summary_item["summary"]
                location_text = summary_item["location"]

                if year_num in processed_years:
                    logger.warning(f"Duplicate year {year_num}. Skipping.")
                    continue
                if not (birth_year <= year_num <= last_year_to_generate):
                    logger.warning(f"Year {year_num} out of range. Skipping.")
                    continue

                summaries_dict[year_num] = (summary_text, location_text, news_context_by_year[year_num])
                processed_years.add(year_num)
        except KeyError as e:
            logger.error(f"Missing key in validated data for year {year}: {e}")
    else:
        logger.error(f"Failed to generate yearly summary for {year}.")

    if not summaries_dict:
        logger.warning(f"No valid yearly summaries processed.")
        return None

    logger.info(f"Processed {len(summaries_dict)} yearly summaries with location and news.")
    return summaries_dict, birth_month, birth_day

async def generate_monthly_summaries_for_year(
    llm_service: LLMService, persona_details_str: str, year: int, yearly_summary: str,
    location_for_year: Optional[str], current_year: int, current_month: int, ddgs_instance: DDGS
) -> Optional[Dict[int, Tuple[str, Optional[str], Optional[str]]]]:
    """Generates monthly summaries with location and includes news context."""
    logger.info(f"Generating monthly summaries for {year} (Loc: {location_for_year})...")
    search_results_summary = "Search placeholder."
    monthly_news_context_by_month = {}
    last_two_months = [(datetime.now().replace(day=1) - timedelta(days=i * 30)).strftime("%Y-%m") for i in range(2)]
    date_string = " to ".join(sorted(last_two_months))
    console.print(date_string) # Example output: ['2025-04', '2025-03']
    # Perform a news search for each of the last two months
    for month_key in last_two_months:
        search_query = f"major world events news {location_for_year or 'the world'} {month_key}"
        logger.info(f"Performing DDGS search for month {month_key}: '{search_query}'")
        search_results_summary = "Search placeholder."
        time.sleep(1)
        try:
            # Perform the search
            search_results_list: List[Dict] = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=5)
            if search_results_list:
                # Format the search results
                formatted_results = [f"- {r.get('title', '')}: {r.get('body', '')[:100]}..." for r in search_results_list]
                search_results_summary = "\n".join(formatted_results)

                # Print the search results using Rich
                console.print(Panel(
                    "\n".join(formatted_results),
                    title=f"News Search Results for {month_key}",
                    border_style="blue",
                    expand=True
                ))
            else:
                search_results_summary = "Search returned no results."
                console.print(Panel(
                    "No results found.",
                    title=f"News Search Results for {month_key}",
                    border_style="red",
                    expand=True
                ))
            logger.debug(f"DDGS monthly context summary for {month_key}:\n{search_results_summary}")
        except Exception as search_err:
            logger.error(f"DDGS search failed for month {month_key}: {search_err}")
            search_results_summary = "Search failed."
            console.print(Panel(
                "Search failed due to an error.",
                title=f"News Search Results for {month_key}",
                border_style="red",
                expand=True
            ))

        # Store the news context for the month
        monthly_news_context_by_month[month_key] = search_results_summary
    # Generate the monthly summaries
    news_context_combined = "\n".join([f"{month_key}: {news}" for month_key, news in monthly_news_context_by_month.items()])
    prompt = f"""
Persona: {persona_details_str}, Year: {year}, Location: {location_for_year or 'inferred'}, Yearly Context: "{yearly_summary}", Search Context: {news_context_combined}
**Task:** Create summaries for the following **MONTHS ONLY** {date_string} of {year}.
**Output Format:** Respond ONLY with JSON: {{"summaries": [{{"month": int, "location": str, "summary": str}}]}}

"""
    validated_data = await _call_llm_and_get_validated_data(llm_service, prompt, MonthlySummariesResponse, f"monthly summaries year {year}")
    if validated_data:
        summaries_list = validated_data["summaries"]
        summaries_dict: Dict[int, Tuple[str, Optional[str], Optional[str]]] = {}
        processed_months = set()
        for item_dict in summaries_list:
            month_num = item_dict["month"]
            summary_text = item_dict["summary"]
            location_text = item_dict["location"]
            if not (1 <= month_num <= 12):
                logger.warning(f"Invalid month {month_num}. Skipping.")
                continue
            if month_num in processed_months:
                logger.warning(f"Duplicate month {month_num}. Skipping.")
                continue

            # Convert the month number to the "YYYY-MM" format to match the keys in monthly_news_context_by_month
            month_key = f"{year}-{month_num:02d}"
            news_context = monthly_news_context_by_month.get(month_key, "")

            summaries_dict[month_num] = (summary_text, location_text, news_context)
            processed_months.add(month_num)
        logger.info(f"Processed {len(summaries_dict)} monthly summaries for {year}.")
        return summaries_dict
    else:
        logger.error(f"Failed validated monthly summaries for {year}.")
        return None

async def generate_daily_summaries_for_month(
    llm_service: LLMService, persona_details_str: str, year: int, month: int, monthly_summary: str,
    yearly_summary: str, location_for_month: Optional[str], current_year: int, current_month: int,
    current_day: int, ddgs_instance: DDGS
) -> Optional[Tuple[Dict[int, Tuple[str, Optional[str], Optional[str]]], str]]:
    """Generates daily summaries for the last 7 days with location and includes news context."""
    logger.info(f"Generating daily summaries for {year}-{month:02d} (Loc: {location_for_month})...")
    daily_news_context_by_day = {}

    # Calculate the last 7 days
    last_7_days = [(datetime.now() - timedelta(days=i)).date() for i in range(7)]
    last_7_days = [d for d in last_7_days if d.year == year and d.month == month]  # Filter for the given month
    if not last_7_days:
        logger.warning(f"No valid days in the last 7 days for {year}-{month:02d}.")
        return None

    # Perform a news search for each day
    for day in last_7_days:
        time.sleep(1)
        day_key = day.strftime("%Y-%m-%d")
        search_query = f"events news {location_for_month or 'region'} {day_key}"
        logger.info(f"Performing DDGS search for day {day_key}: '{search_query}'")
        search_results_summary = "Search placeholder."
        try:
            search_results_list: List[Dict] = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=5)
            if search_results_list:
                formatted_results = [f"- {r.get('title', '')}: {r.get('body', '')[:100]}..." for r in search_results_list]
                search_results_summary = "\n".join(formatted_results)
                console.print(Panel(
                    "\n".join(formatted_results),
                    title=f"News Search Results for {day_key}",
                    border_style="blue",
                    expand=True
                ))
            else:
                search_results_summary = "Search returned no results."
                console.print(Panel(
                    "No results found.",
                    title=f"News Search Results for {day_key}",
                    border_style="red",
                    expand=True
                ))
            logger.debug(f"DDGS daily context summary for {day_key}:\n{search_results_summary}")
        except Exception as search_err:
            logger.error(f"DDGS search failed for day {day_key}: {search_err}")
            search_results_summary = "Search failed."
            console.print(Panel(
                "Search failed due to an error.",
                title=f"News Search Results for {day_key}",
                border_style="red",
                expand=True
            ))

        # Store the news context for the day
        daily_news_context_by_day[day.day] = search_results_summary

    # Combine all daily news contexts into one string
    news_context_combined = "\n".join([f"{day}: {news}" for day, news in daily_news_context_by_day.items()])

    # Generate the daily summaries
    prompt = f"""
Persona: {persona_details_str}, Context: Year={year}("{yearly_summary}"), Month={month}("{monthly_summary}"), Location: {location_for_month or 'inferred'}, News Context: {news_context_combined}
**Task:** Generate summaries for the following **DAYS ONLY** {', '.join([str(d.day) for d in last_7_days])} of {month}, {year}.
**Output Format:** Respond ONLY with JSON: {{"summaries": [{{"day": int, "location": str, "summary": str}}]}}

"""
    validated_data = await _call_llm_and_get_validated_data(llm_service, prompt, DailySummariesResponse, f"daily summaries {year}-{month:02d}")
    if validated_data:
        summaries_list = validated_data["summaries"]
        summaries_dict: Dict[int, Tuple[str, Optional[str], Optional[str]]] = {}
        processed_days = set()
        for item_dict in summaries_list:
            day_num = item_dict["day"]
            summary_text = item_dict["summary"]
            location_text = item_dict["location"]
            if not (1 <= day_num <= 31):
                logger.warning(f"Invalid day {day_num}. Skipping.")
                continue
            if day_num in processed_days:
                logger.warning(f"Duplicate day {day_num}. Skipping.")
                continue
            summaries_dict[day_num] = (summary_text, location_text, daily_news_context_by_day.get(day_num, ""))
            processed_days.add(day_num)
        logger.info(f"Processed {len(summaries_dict)} daily summaries for {year}-{month:02d}.")
        return summaries_dict
    else:
        logger.error(f"Failed validated daily summaries for {year}-{month:02d}.")
        return None

async def generate_hourly_breakdown_for_day(
    llm_service: LLMService, persona_details_str: str, year: int, month: int, day: int, daily_summary: str,
    monthly_summary: str, yearly_summary: str, location_for_day: Optional[str], current_year: int,
    current_month: int, current_day: int, current_hour: int, ddgs_instance: DDGS
) -> Optional[Dict[int, Tuple[str, Optional[str]]]]:
    """Generates hourly breakdown with location and includes news searches."""
    logger.info(f"Generating hourly breakdown for {year}-{month:02d}-{day:02d} (Loc: {location_for_day})...")
    search_query = f"events news {location_for_day or 'region'} {year}-{month:02d}-{day:02d}"
    logger.info(f"Performing DDGS search: '{search_query}'")
    search_results_summary = "Search placeholder."
    try:
        search_results_list: List[Dict] = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=5)
        if search_results_list:
            formatted_results = [f"- {r.get('title', '')}: {r.get('body', '')[:100]}..." for r in search_results_list]
            search_results_summary = "\n".join(formatted_results)
            console.print(Panel(
                "\n".join(formatted_results),
                title=f"News Search Results for {year}-{month:02d}-{day:02d}",
                border_style="blue",
                expand=True
            ))
        else:
            search_results_summary = "Search returned no results."
            console.print(Panel(
                "No results found.",
                title=f"News Search Results for {year}-{month:02d}-{day:02d}",
                border_style="red",
                expand=True
            ))
        logger.debug(f"DDGS daily context summary:\n{search_results_summary}")
    except Exception as search_err:
        logger.error(f"DDGS search failed: {search_err}")
        search_results_summary = "Search failed."
        console.print(Panel(
            "Search failed due to an error.",
            title=f"News Search Results for {year}-{month:02d}-{day:02d}",
            border_style="red",
            expand=True
        ))

    # Generate the hourly breakdown
    prompt = f"""
Persona: {persona_details_str}, Date: {year}-{month:02d}-{day:02d}, Location: {location_for_day or 'inferred'}
Context: Year="{yearly_summary}", Month="{monthly_summary}", Day="{daily_summary}", Search="{search_results_summary}"
**Task:** Generate hour-by-hour breakdown (0-23).
**Output Format:** Respond ONLY with JSON: {{"activities": [{{"hour": int, "location": str, "activity": str}}]}}

"""
    validated_data = await _call_llm_and_get_validated_data(llm_service, prompt, HourlyBreakdownResponse, f"hourly breakdown {year}-{month:02d}-{day:02d}")
    if validated_data:
        activities_list = validated_data["activities"]
        activities_dict: Dict[int, Tuple[str, Optional[str]]] = {}
        processed_hours = set()
        for item_dict in activities_list:
            hour_num = item_dict["hour"]
            activity_text = item_dict["activity"]
            location_text = item_dict["location"]
            if not (0 <= hour_num <= 23):
                logger.warning(f"Invalid hour {hour_num}. Skipping.")
                continue
            if hour_num in processed_hours:
                logger.warning(f"Duplicate hour {hour_num}. Skipping.")
                continue
            activities_dict[hour_num] = (activity_text, location_text)
            processed_hours.add(hour_num)
        if not activities_dict:
            logger.warning(f"No valid hourly activities for {year}-{month:02d}-{day:02d}.")
            return None
        if year == current_year and month == current_month and day == current_day:
            hours_to_remove = [h for h in activities_dict if h > current_hour]
            if hours_to_remove:
                logger.info(f"Pruning future hours for {year}-{month:02d}-{day:02d}.")
                for h in hours_to_remove:
                    del activities_dict[h]
        logger.info(f"Processed {len(activities_dict)} hourly activities for {year}-{month:02d}-{day:02d}.")
        return {"activities": activities_dict, "news": search_results_summary}
    else:
        logger.error(f"Failed validated hourly activities for {year}-{month:02d}-{day:02d}.")
        return None

# <<< IMPORTANT: Define generate_life_summary_sequentially BEFORE generate_new_simulacra_background >>>
async def generate_life_summary_sequentially(
    llm_service: LLMService,
    persona_details: Dict[str, Any],
    age: int
) -> Optional[Dict[str, Any]]:
    """
    Orchestrates generation with simplified loop logic, calling the original generator functions.
    Updates the summary_tree only after life_summary is fully generated.
    """
    # --- Setup: Date/Time and Persona Info ---
    now = datetime.now(timezone.utc) # Use timezone aware datetime
    today = now.date()
    current_year = now.year
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    console.print(f"Generation started: [cyan]{now.strftime('%Y-%m-%d %H:%M:%S %Z')}[/cyan]")

    # --- Birthdate Handling ---
    birthdate_str = persona_details.get("Birthdate")
    birthdate = None
    if birthdate_str:
        try:
            birthdate = datetime.strptime(birthdate_str, "%Y-%m-%d").date()
        except ValueError:
            logger.warning(f"Invalid Birthdate format '{birthdate_str}'. Estimating.")
            birthdate_str = None # Clear invalid string to trigger estimation

    if not birthdate_str:
        # Estimate if missing or invalid
        birth_year_approx = current_year - persona_details.get("Age", age) # Use Age from persona first
        # Simple estimation, consider refining (e.g., random month/day)
        birthdate = date(birth_year_approx, 7, 1) # Default to mid-year
        logger.warning(f"Birthdate missing or invalid. Using estimated: {birthdate.strftime('%Y-%m-%d')}")
        persona_details["Birthdate"] = birthdate.strftime("%Y-%m-%d") # Update persona details
    elif birthdate is None: # Should not happen if logic above is correct, but safety check
         logger.error("Failed to determine birthdate. Cannot proceed.")
         return None


    birth_year = birthdate.year
    birth_month = birthdate.month
    birth_day = birthdate.day
    actual_age = today.year - birth_year - ((today.month, today.day) < (birth_month, birth_day))
    logger.info(f"Persona: {persona_details.get('Name', 'N/A')}, Birthdate: {birthdate}, Actual Age: {actual_age}")
    persona_details["Age"] = actual_age # Ensure Age field matches calculated age

    # --- Initialize main data structure ---
    life_summary = {
        "persona_details": persona_details,
        "generation_info": {
            "generated_at": now.isoformat(),
            "current_year": current_year,
            "current_month": current_month,
            "current_day": current_day,
            "current_hour": current_hour,
        },
        "birth_year": birth_year,
        "birth_month": birth_month,
        "birth_day": birth_day,
        "initial_relationships": None,
        "yearly_summaries": {},
        "monthly_summaries": {},
        "daily_summaries": {},
        "hourly_breakdowns": {}
    }

    # --- Persona Details String & DDGS Init ---
    details_list = [f"{k}: {v}" for k, v in persona_details.items()]
    persona_details_str = ", ".join(details_list)
    ddgs_search_tool = DDGS()

    # --- Step 0: Generate Initial Relationships ---
    console.print(Rule("Generating Initial Relationships", style="bold yellow"))
    try:
        initial_relationships_data = await generate_initial_relationships(llm_service, persona_details_str)
        if initial_relationships_data:
            life_summary["initial_relationships"] = initial_relationships_data
    except Exception as e:
        logger.error(f"Error generating initial relationships: {e}", exc_info=True)

    # --- Step 1: Generate Yearly Summaries ---
    console.print(Rule("Generating Yearly Summaries (All Years)", style="bold yellow"))
    try:
        yearly_result = await generate_yearly_summaries(
            llm_service, persona_details_str,
            json.dumps(life_summary["initial_relationships"] or {}),
            birth_year, current_year, current_year, # Generate up to current year
            ddgs_instance=ddgs_search_tool
        )
        if yearly_result:
            yearly_summaries_tuples, bm_returned, bd_returned = yearly_result
            # Update birth month/day if LLM provided them for the birth year
            if bm_returned and bd_returned and not (life_summary["birth_month"] and life_summary["birth_day"]):
                 life_summary["birth_month"] = bm_returned
                 life_summary["birth_day"] = bd_returned
                 logger.info(f"Updated birth month/day from yearly summary: {bm_returned}/{bd_returned}")

            for year, (summary, location, news) in yearly_summaries_tuples.items():
                life_summary["yearly_summaries"][year] = {
                    "summary": summary, "location": location, "news": news
                }
    except Exception as e:
        logger.error(f"Error generating yearly summaries: {e}", exc_info=True)

    # --- Step 2: Generate Monthly Summaries ---
    console.print(Rule("Generating Monthly Summaries (Current & Previous Month)", style="bold yellow"))
    try:
        # Determine the months to generate (current and previous)
        months_to_generate = []
        current_date = date(current_year, current_month, 1)
        months_to_generate.append((current_date.year, current_date.month))
        # Previous month calculation
        if current_date.month == 1:
            prev_month_date = date(current_date.year - 1, 12, 1)
        else:
            prev_month_date = date(current_date.year, current_date.month - 1, 1)
        # Only generate previous month if it's after birth year
        if prev_month_date.year >= birth_year:
             months_to_generate.append((prev_month_date.year, prev_month_date.month))

        for year, month in months_to_generate:
            yearly_context = life_summary["yearly_summaries"].get(year, {})
            # <<< FIX: Added await >>>
            monthly_result = await generate_monthly_summaries_for_year(
                llm_service, persona_details_str, year,
                yearly_context.get("summary", "No yearly summary available."),
                yearly_context.get("location"),
                current_year, current_month,
                ddgs_instance=ddgs_search_tool
            )
            if monthly_result:
                life_summary["monthly_summaries"].setdefault(year, {}).update({
                    m: {"summary": summary, "location": location, "news": news}
                    for m, (summary, location, news) in monthly_result.items()
                    # Only add if the month 'm' is one we intended to generate
                    if (year, m) in months_to_generate
                })
    except Exception as e:
        logger.error(f"Error generating monthly summaries: {e}", exc_info=True)

    # --- Step 3: Generate Daily Summaries ---
    console.print(Rule("Generating Daily Summaries (Last 7 Days)", style="bold yellow"))
    try:
        days_to_generate = [(today - timedelta(days=i)) for i in range(7)]
        # Filter days to be on or after birthdate
        days_to_generate = [d for d in days_to_generate if d >= birthdate]

        for day_date in days_to_generate:
            year, month, day_num = day_date.year, day_date.month, day_date.day
            monthly_context = life_summary["monthly_summaries"].get(year, {}).get(month, {})
            yearly_context_summary = life_summary["yearly_summaries"].get(year, {}).get("summary", "No yearly summary.")
            # <<< FIX: Added await >>>
            daily_result = await generate_daily_summaries_for_month(
                llm_service, persona_details_str, year, month,
                monthly_context.get("summary", "No monthly summary available."),
                yearly_context_summary,
                monthly_context.get("location"), # Use month's location as context
                current_year, current_month, current_day,
                ddgs_instance=ddgs_search_tool
            )
            if daily_result:
                 # daily_result is Dict[int, Tuple[str, Optional[str], Optional[str]]]
                life_summary["daily_summaries"].setdefault(year, {}).setdefault(month, {}).update({
                    d: {"summary": summary, "location": location, "news": news}
                    for d, (summary, location, news) in daily_result.items()
                    # Only add if the day 'd' is one we intended to generate
                    if date(year, month, d) in days_to_generate
                })
    except Exception as e:
        logger.error(f"Error generating daily summaries: {e}", exc_info=True)

    # --- Step 4: Generate Hourly Breakdowns ---
    console.print(Rule("Generating Hourly Breakdowns (Today & Yesterday)", style="bold yellow"))
    try:
        days_for_hourly = [today, today - timedelta(days=1)]
        # Filter days to be on or after birthdate
        days_for_hourly = [d for d in days_for_hourly if d >= birthdate]

        for day_date in days_for_hourly:
            year, month, day_num = day_date.year, day_date.month, day_date.day
            daily_context = life_summary["daily_summaries"].get(year, {}).get(month, {}).get(day_num, {})
            monthly_context_summary = life_summary["monthly_summaries"].get(year, {}).get(month, {}).get("summary", "No monthly summary.")
            yearly_context_summary = life_summary["yearly_summaries"].get(year, {}).get("summary", "No yearly summary.")
            # <<< FIX: Added await >>>
            hourly_result = await generate_hourly_breakdown_for_day(
                llm_service, persona_details_str, year, month, day_num,
                daily_context.get("summary", "No daily summary available."),
                monthly_context_summary,
                yearly_context_summary,
                daily_context.get("location"), # Use day's location as context
                current_year, current_month, current_day, current_hour,
                ddgs_instance=ddgs_search_tool
            )
            if hourly_result:
                # hourly_result is {"activities": Dict[int, Tuple[str, Optional[str]]], "news": str}
                # Ensure structure matches expected format
                hourly_data_to_store = {
                     "activities": hourly_result.get("activities", {}),
                     "news": hourly_result.get("news")
                }
                life_summary["hourly_breakdowns"].setdefault(year, {}).setdefault(month, {}).setdefault(day_num, hourly_data_to_store)

    except Exception as e:
        logger.error(f"Error generating hourly breakdowns: {e}", exc_info=True)

    # --- Final Output: Build and Print Summary Tree ---
    summary_tree = Tree(f"[bold blue]Life Summary for {persona_details.get('Name', 'Unknown')}[/bold blue]")
    # --- Persona Details ---
    persona_node = summary_tree.add(f"[bold green]Persona Details[/bold green]")
    for key, value in persona_details.items():
        persona_node.add(f"{key}: {value}")
    # --- Birth Information ---
    birth_info_node = summary_tree.add(f"[bold green]Birth Information[/bold green]")
    birth_info_node.add(f"Year: {birth_year}")
    birth_info_node.add(f"Month: {birth_month}")
    birth_info_node.add(f"Day: {birth_day}")
    # --- Relationships ---
    relationships_node = summary_tree.add(f"[bold green]Initial Relationships[/bold green]")
    if life_summary["initial_relationships"]:
        parents = life_summary["initial_relationships"].get("parents", [])
        siblings = life_summary["initial_relationships"].get("siblings", [])
        for parent in parents:
            relationships_node.add(f"Parent: {parent.get('name', 'Unknown')}")
        for sibling in siblings:
            relationships_node.add(f"Sibling: {sibling.get('name', 'Unknown')}")
    # --- Yearly Summaries ---
    yearly_summaries_node = summary_tree.add(f"[bold green]Yearly Summaries[/bold green]")
    for year, data in life_summary["yearly_summaries"].items():
        yearly_summaries_node.add(f"[bold yellow]{year}[/bold yellow]: {data.get('summary', 'No summary')}")
    # --- Monthly Summaries ---
    monthly_summaries_node = summary_tree.add(f"[bold green]Monthly Summaries[/bold green]")
    for year, months in life_summary["monthly_summaries"].items():
        year_node = monthly_summaries_node.add(f"[bold yellow]{year}[/bold yellow]")
        for month, data in months.items():
            year_node.add(f"[bold yellow]{month}[/bold yellow]: {data.get('summary', 'No summary')}")
    # --- Daily Summaries ---
    daily_summaries_node = summary_tree.add(f"[bold green]Daily Summaries[/bold green]")
    for year, months in life_summary["daily_summaries"].items():
        year_node = daily_summaries_node.add(f"[bold yellow]{year}[/bold yellow]")
        for month, days in months.items():
            month_node = year_node.add(f"[bold yellow]{month}[/bold yellow]")
            for day, data in days.items():
                month_node.add(f"[bold yellow]{day}[/bold yellow]: {data.get('summary', 'No summary')}")
    # --- Hourly Breakdowns ---
    hourly_breakdowns_node = summary_tree.add(f"[bold green]Hourly Breakdowns[/bold green]")
    for year, months in life_summary["hourly_breakdowns"].items():
        year_node = hourly_breakdowns_node.add(f"[bold yellow]{year}[/bold yellow]")
        for month, days in months.items():
            month_node = year_node.add(f"[bold yellow]{month}[/bold yellow]")
            for day, data in days.items():
                day_node = month_node.add(f"[bold yellow]{day}[/bold yellow]")
                for hour, activity in data.get("activities", {}).items():
                    day_node.add(f"[bold yellow]{hour}:00[/bold yellow] - {activity[0]} at {activity[1]}")
    console.print(Rule("Generation Complete", style="bold green"))
    console.print(summary_tree)

    return life_summary

# <<< This function definition MUST come AFTER all the functions it calls >>>
async def generate_new_simulacra_background(
    sim_id: str,
    world_instance_uuid: Optional[str],
    world_type: str,
    world_description: str,
    age_range: Tuple[int, int] = (18, 45) # age_range is used by generate_random_persona implicitly via prompt
) -> Optional[Dict[str, Any]]:
    """Generates a new persona and their life summary using the simplified structure."""

    # Check for required IDs
    if not world_instance_uuid:
        logger.error("World instance UUID is required for generating background. Aborting.")
        console.print("[bold red]Error: World instance UUID missing, cannot generate persona.[/bold red]")
        return None
    if not sim_id:
        logger.error("Simulacra ID is required for generating background. Aborting.")
        console.print("[bold red]Error: Simulacra ID missing, cannot generate persona.[/bold red]")
        return None

    logger.info(f"Starting new Simulacra background generation for sim: {sim_id} (UUID: {world_instance_uuid})...")
    logger.info(f"World Context: Type='{world_type}', Description='{world_description[:100]}...'")

    # --- LLM Initialization ---
    api_key = os.getenv("GOOGLE_API_KEY")
    llm_service = None
    try:
        llm_service = LLMService(api_key=api_key)
        logger.info("LLMService initialized for generation.")
    except Exception as e:
        logger.critical(f"Error initializing LLMService: {e}. Cannot proceed.", exc_info=True)
        console.print(Panel(f"[bold red]Fatal Error:[/bold red] Could not initialize LLM Service: {e}", title="Initialization Failed", border_style="red"))
        return None

    # --- Generate Random Persona ---
    console.print(Rule(f"Generating Random Persona for '{world_type}' world", style="bold yellow"))
    generated_persona = await generate_random_persona(
        llm_service=llm_service,
        world_type=world_type,
        world_description=world_description
        # age_range is implicitly handled by the prompt inside generate_random_persona
    )

    # --- Fallback and Birthdate Handling ---
    if not generated_persona:
        logger.error("Could not generate persona via LLM. Using fallback.")
        fallback_age = 30 # Example fallback age
        fallback_birth_year = datetime.now().year - fallback_age
        generated_persona = {
            "Name": "Alex Default", "Age": fallback_age, "Occupation": "Archivist",
            "Current_location": "Default City, Default State", "Personality_Traits": ["Methodical", "Calm", "Inquisitive"],
            "Birthplace": "Old Town, Default State", "Education": "Degree in History",
            "Birthdate": f"{fallback_birth_year}-07-15", # Example fixed birthdate
            "Physical_Appearance": "Average height, tidy appearance."
        }
        console.print(Panel(pretty_repr(generated_persona), title="Using Fallback Persona", border_style="red", expand=False))
    else:
         # Ensure birthdate exists and is valid, estimate if missing/invalid
        birthdate_str = generated_persona.get("Birthdate")
        valid_birthdate = False
        if birthdate_str and re.match(r"\d{4}-\d{2}-\d{2}", str(birthdate_str)):
            try:
                datetime.strptime(str(birthdate_str), "%Y-%m-%d")
                valid_birthdate = True
            except ValueError:
                logger.warning(f"Generated persona has invalid date format '{birthdate_str}'. Estimating.")

        if not valid_birthdate:
             logger.warning("Generated persona missing valid 'Birthdate'. Estimating.")
             age_from_persona = generated_persona.get("Age", 30) # Use LLM age if available
             birth_year_est = datetime.now().year - age_from_persona
             # Consider a more robust default/estimation (e.g., random month/day)
             generated_persona["Birthdate"] = f"{birth_year_est}-07-15"

        console.print(Panel(pretty_repr(generated_persona), title="Generated Persona", border_style="blue", expand=False))

    # Use age from persona (potentially updated/calculated)
    current_age = generated_persona.get("Age", 30)

    # --- Run Life Summary Generation ---
    console.print(Rule(f"Generating Life Summary (Birthdate: {generated_persona.get('Birthdate')})", style="bold yellow"))
    # <<< This call MUST happen AFTER generate_life_summary_sequentially is defined >>>
    life_data = await generate_life_summary_sequentially(
        llm_service=llm_service,
        persona_details=generated_persona, # Pass the potentially updated persona
        age=current_age # Pass age, primarily for initial estimation if birthdate fails
    )

    if not life_data:
        logger.error("Life summary generation failed critically.")
        console.print(Panel("[bold red]Error:[/bold red] Life summary generation failed.", title="Generation Failed", border_style="red"))
        return None # Return None if the core generation fails

    # --- Add Metadata and Save Results ---
    life_data["simulacra_id"] = sim_id
    life_data["world_instance_uuid"] = world_instance_uuid
    if "generation_info" in life_data:
        life_data["generation_info"]["generated_at"] = datetime.now(timezone.utc).isoformat()
    else:
        logger.warning("life_data missing 'generation_info', cannot update timestamp.")


    # --- Save Results ---
    persona_details_final = life_data.get("persona_details", {})
    persona_name_safe = re.sub(r'[^\w\-]+', '_', persona_details_final.get('Name', 'Unknown'))
    persona_name_safe = persona_name_safe[:30] # Limit length for filename safety
    # Use the world_instance_uuid for uniqueness per instance
    output_filename = f"life_summary_{persona_name_safe}_{world_instance_uuid}.json"
    output_path = os.path.join(LIFE_SUMMARY_DIR, output_filename)

    console.print(Rule(f"Saving results to [green]{output_path}[/green]", style="bold green"))
    try:
        os.makedirs(LIFE_SUMMARY_DIR, exist_ok=True) # Ensure directory exists
        with open(output_path, 'w', encoding='utf-8') as f:
            # Use default=str for any non-serializable types like datetime objects if they sneak in
            json.dump(life_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved successfully to {output_path}.")
        console.print(f"[bold green]✔ Results saved successfully to {output_path}[/bold green]")
    except Exception as e:
        logger.error(f"Error saving results to {output_path}: {e}", exc_info=True)
        console.print(Panel(f"[bold red]Error saving results:[/bold red] {e}", title="Save Failed", border_style="red"))
        # Decide if returning data despite save failure is okay (current code does)

    # Final check if persona_details key exists before returning
    if "persona_details" not in life_data:
        logger.error("Generated life_data is missing the 'persona_details' key before returning.")
        # Consider returning None here if this is critical
        # return None

    return life_data