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
from datetime import date, timedelta, datetime
from typing import Dict, Any, Optional, List, Tuple, Type, TypeVar, Union
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

async def generate_random_persona(llm_service: LLMService) -> Optional[Dict[str, Any]]:
    """Generates a random plausible persona including age using the LLM."""
    logger.info("Generating random persona...")
    prompt = f"""
Create a detailed and plausible random fictional persona profile.
**Required Fields:** Name, Age (18-45 integer), Occupation, Current_location (City, State/Country), Personality_Traits (list, 3-6 adjectives), Birthplace (City, State/Country), Education, Physical_Appearance (brief description).
Respond ONLY with valid JSON matching the required fields.

"""
    validated_data = await _call_llm_and_get_validated_data(llm_service, prompt, PersonaDetailsResponse, "random persona generation")
    if validated_data:
        logger.info(f"Successfully generated random persona: {validated_data.get('Name', 'Unknown')}, Age: {validated_data.get('Age', 'N/A')}")
        # console.print(Panel(pretty_repr(validated_data), title="Validated Persona Data", border_style="green", expand=True))
        return validated_data
    else:
        logger.error("Failed to generate or validate random persona.")
        return None

async def generate_life_summary_sequentially(
    llm_service: LLMService,
    persona_details: Dict[str, Any],
    age: int  # Keep age for initial calculation/display
) -> Optional[Dict[str, Any]]:
    """
    Orchestrates generation with simplified loop logic, calling the original generator functions.
    Updates the summary_tree only after life_summary is fully generated.
    """
    # --- Setup: Date/Time and Persona Info (Same as before) ---
    now = datetime.now()
    today = now.date()
    current_year = now.year
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    console.print(f"Generation started: [cyan]{now.strftime('%Y-%m-%d %H:%M:%S')}[/cyan]")

    # --- Birthdate Handling (Same as before) ---
    birthdate_str = persona_details.get("Birthdate")
    if not birthdate_str:
        birth_year_approx = current_year - age
        birthdate = date(birth_year_approx, 1, 1)
        logger.warning(f"Birthdate missing from persona. Using estimated: {birthdate.strftime('%Y-%m-%d')}")
        persona_details["Birthdate"] = birthdate.strftime("%Y-%m-%d")
    else:
        try:
            birthdate = datetime.strptime(birthdate_str, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid Birthdate format '{birthdate_str}'. Needs YYYY-MM-DD. Cannot proceed.")
            return None
    birth_year = birthdate.year
    birth_month = birthdate.month
    birth_day = birthdate.day
    actual_age = today.year - birth_year - ((today.month, today.day) < (birth_month, birth_day))
    logger.info(f"Persona: {persona_details.get('Name', 'N/A')}, Birthdate: {birthdate}, Actual Age: {actual_age}")

    # --- Initialize main data structure (Same as before) ---
    life_summary = {
        "persona_details": persona_details,
        "generation_info": {  # Meta info about the generation run
            "generated_at": now.isoformat(),
            "current_year": current_year,
            "current_month": current_month,
            "current_day": current_day,
            "current_hour": current_hour,
        },
        "birth_year": birth_year,  # Store determined birth info
        "birth_month": birth_month,
        "birth_day": birth_day,
        "initial_relationships": None,
        "yearly_summaries": {},    # Format: { year: {"summary": str, "location": str|None, "news": str|None} }
        "monthly_summaries": {},   # Format: { year: { month: {"summary": str, "location": str|None, "news": str|None} } }
        "daily_summaries": {},     # Format: { year: { month: { day: {"summary": str, "location": str|None, "news": str|None} } } }
        "hourly_breakdowns": {}    # Format: { year: { month: { day: { hour: {"activity": str, "location": str|None, "news": str|None} } } } }
    }

    # --- Persona Details String & DDGS Init (Same as before) ---
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
            birth_year, current_year, current_year,
            ddgs_instance=ddgs_search_tool
        )
        if yearly_result:
            yearly_summaries_tuples, bm_returned, bd_returned = yearly_result
            for year, (summary, location, news) in yearly_summaries_tuples.items():
                life_summary["yearly_summaries"][year] = {
                    "summary": summary, "location": location, "news": news
                }
    except Exception as e:
        logger.error(f"Error generating yearly summaries: {e}", exc_info=True)

    # --- Step 2: Generate Monthly Summaries ---
    console.print(Rule("Generating Monthly Summaries (Current & Previous Month)", style="bold yellow"))
    try:
        months_to_generate = [
            (current_year, current_month),
            (current_year, current_month - 1) if current_month > 1 else (current_year - 1, 12)
        ]
        for year, month in months_to_generate:
            yearly_context = life_summary["yearly_summaries"].get(year, {})
        monthly_result = await generate_monthly_summaries_for_year(
            llm_service, persona_details_str, year,
            yearly_context.get("summary", ""),
            yearly_context.get("location"),
            current_year, current_month,
            ddgs_instance=ddgs_search_tool
        )
        if monthly_result:
            life_summary["monthly_summaries"].setdefault(year, {}).update({
                month: {"summary": summary, "location": location, "news": news}
                for month, (summary, location, news) in monthly_result.items()
            })
    except Exception as e:
        logger.error(f"Error generating monthly summaries: {e}", exc_info=True)

    # --- Step 3: Generate Daily Summaries ---
    console.print(Rule("Generating Daily Summaries (Last 7 Days)", style="bold yellow"))
    try:
        days_to_generate = [(today - timedelta(days=i)) for i in range(7)]
        for day in days_to_generate:
            year, month, day_num = day.year, day.month, day.day
            monthly_context = life_summary["monthly_summaries"].get(year, {}).get(month, {})
        daily_result = await generate_daily_summaries_for_month(
            llm_service, persona_details_str, year, month,
            monthly_context.get("summary", ""),
            life_summary["yearly_summaries"].get(year, {}).get("summary", ""),
            monthly_context.get("location"),
            current_year, current_month, current_day,
            ddgs_instance=ddgs_search_tool
        )
        if daily_result:
            life_summary["daily_summaries"].setdefault(year, {}).setdefault(month, {}).update({
                day: {"summary": summary, "location": location, "news": news}
                for day, (summary, location, news) in daily_result.items()
            })
    except Exception as e:
        logger.error(f"Error generating daily summaries: {e}", exc_info=True)

    # --- Step 4: Generate Hourly Breakdowns ---
    console.print(Rule("Generating Hourly Breakdowns (Today & Yesterday)", style="bold yellow"))
    try:
        for day in [today, today - timedelta(days=1)]:
            year, month, day_num = day.year, day.month, day.day
            daily_context = life_summary["daily_summaries"].get(year, {}).get(month, {}).get(day_num, {})
            hourly_result = await generate_hourly_breakdown_for_day(
                llm_service, persona_details_str, year, month, day_num,
                daily_context.get("summary", ""),
                life_summary["monthly_summaries"].get(year, {}).get(month, {}).get("summary", ""),
                life_summary["yearly_summaries"].get(year, {}).get("summary", ""),
                daily_context.get("location"),
                current_year, current_month, current_day, current_hour,
                ddgs_instance=ddgs_search_tool
            )
            if hourly_result:
                life_summary["hourly_breakdowns"].setdefault(year, {}).setdefault(month, {}).setdefault(day_num, {}).update(hourly_result)
    except Exception as e:
        logger.error(f"Error generating hourly breakdowns: {e}", exc_info=True)

    # --- Final Output: Build and Print Summary Tree ---
    summary_tree = Tree(f"[bold blue]Life Summary for {persona_details.get('Name', 'Unknown')}[/bold blue]")

    # Add Persona Details
    persona_node = summary_tree.add("[green]Persona Details[/green]")
    for key, value in life_summary["persona_details"].items():
        persona_node.add(f"[cyan]{key}[/cyan]: {value}")

    # Add Initial Relationships
    relationships_node = summary_tree.add("[green]Initial Relationships[/green]")
    if life_summary["initial_relationships"]:
        for rel_type, rel_list in life_summary["initial_relationships"].items():
            rel_node = relationships_node.add(f"[cyan]{rel_type.capitalize()}[/cyan]")
            for rel in rel_list:
                rel_node.add(pretty_repr(rel))
    else:
        relationships_node.add("[yellow]No relationships generated.[/yellow]")

    # Add Yearly Summaries
    yearly_node = summary_tree.add("[green]Yearly Summaries[/green]")
    if life_summary["yearly_summaries"]:
        for year, summary in life_summary["yearly_summaries"].items():
            year_node = yearly_node.add(f"[cyan]{year}[/cyan]")
            year_node.add(f"[bold]Summary:[/bold] {summary['summary']}")
            year_node.add(f"[bold]Location:[/bold] {summary.get('location', 'N/A')}")
            year_node.add(f"[bold]News:[/bold] {summary.get('news', 'N/A')}")
    else:
        yearly_node.add("[yellow]No yearly summaries generated.[/yellow]")

    # Add Monthly Summaries
    monthly_node = summary_tree.add("[green]Monthly Summaries[/green]")
    if life_summary["monthly_summaries"]:
        for year, months in life_summary["monthly_summaries"].items():
            year_month_node = monthly_node.add(f"[cyan]{year}[/cyan]")
            for month, summary in months.items():
                month_node = year_month_node.add(f"[cyan]{calendar.month_name[month]}[/cyan]")
                month_node.add(f"[bold]Summary:[/bold] {summary['summary']}")
                month_node.add(f"[bold]Location:[/bold] {summary.get('location', 'N/A')}")
                month_node.add(f"[bold]News:[/bold] {summary.get('news', 'N/A')}")
    else:
        monthly_node.add("[yellow]No monthly summaries generated.[/yellow]")

    # Add Daily Summaries
    daily_node = summary_tree.add("[green]Daily Summaries[/green]")
    if life_summary["daily_summaries"]:
        for year, months in life_summary["daily_summaries"].items():
            year_daily_node = daily_node.add(f"[cyan]{year}[/cyan]")
            for month, days in months.items():
                month_daily_node = year_daily_node.add(f"[cyan]{calendar.month_name[month]}[/cyan]")
                for day, summary in days.items():
                    day_node = month_daily_node.add(f"[cyan]Day {day}[/cyan]")
                    day_node.add(f"[bold]Summary:[/bold] {summary['summary']}")
                    day_node.add(f"[bold]Location:[/bold] {summary.get('location', 'N/A')}")
                    day_node.add(f"[bold]News:[/bold] {summary.get('news', 'N/A')}")
    else:
        daily_node.add("[yellow]No daily summaries generated.[/yellow]")

    # Add Hourly Breakdowns
    hourly_node = summary_tree.add("[green]Hourly Breakdowns[/green]")
    if life_summary["hourly_breakdowns"]:
        for year, months in life_summary["hourly_breakdowns"].items():
            year_hourly_node = hourly_node.add(f"[cyan]{year}[/cyan]")
            for month, days in months.items():
                month_hourly_node = year_hourly_node.add(f"[cyan]{calendar.month_name[month]}[/cyan]")
                for day, hours in days.items():
                    day_hourly_node = month_hourly_node.add(f"[cyan]Day {day}[/cyan]")
                    for hour, activity in hours["activities"].items():
                        hour_node = day_hourly_node.add(f"[cyan]{hour}:00[/cyan]")
                        hour_node.add(f"[bold]Activity:[/bold] {activity[0]}")
                        hour_node.add(f"[bold]Location:[/bold] {activity[1]}")
    else:
        hourly_node.add("[yellow]No hourly breakdowns generated.[/yellow]")

    # Print the summary_tree
    console.print(Rule("Generation Complete", style="bold green"))
    console.print(summary_tree)

    return life_summary

async def generate_new_simulacra_background(age_range: Tuple[int, int] = (18, 45)) -> Optional[Dict[str, Any]]:
    """Generates a new persona and their life summary using the simplified structure."""
    # ... (LLM Init, Persona Gen, Call generate_life_summary_sequentially, Save Results) ...
    # [This code block is identical to the previous version]
    logger.info("Starting new Simulacra background generation...")
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

    # --- Generate Random Persona (Ensure Birthdate is included/estimated) ---
    console.print(Rule("Generating Random Persona", style="bold yellow"))
    generated_persona = await generate_random_persona(llm_service)

    if not generated_persona:
        logger.error("Could not generate persona. Using fallback.")
        fallback_birth_year = datetime.now().year - 30
        generated_persona = {
            "Name": "Alex Default", "Age": 30, "Occupation": "Archivist",
            "Current_location": "Default City, Default State", "Personality_Traits": ["Methodical", "Calm", "Inquisitive"],
            "Birthplace": "Old Town, Default State", "Education": "Degree in History",
            "Birthdate": f"{fallback_birth_year}-07-15", # Example fixed birthdate
            "Physical_Appearance": "Average height, tidy appearance."
        }
        console.print(Panel(pretty_repr(generated_persona), title="Using Fallback Persona", border_style="red", expand=False))
    else:
        # Ensure birthdate exists, generate if missing
        if "Birthdate" not in generated_persona or not re.match(r"\d{4}-\d{2}-\d{2}", str(generated_persona.get("Birthdate"))):
             logger.warning("Generated persona missing valid 'Birthdate'. Estimating.")
             age_from_persona = generated_persona.get("Age", 30)
             birth_year_est = datetime.now().year - age_from_persona
             generated_persona["Birthdate"] = f"{birth_year_est}-07-15"

        console.print(Panel(pretty_repr(generated_persona), title="Generated Persona", border_style="blue", expand=False))

    current_age = generated_persona.get("Age", 30) # Get age for display/initial info

    # --- Run Life Summary Generation ---
    console.print(Rule(f"Generating Life Summary (Birthdate: {generated_persona.get('Birthdate')})", style="bold yellow"))
    life_data = await generate_life_summary_sequentially(
        llm_service=llm_service,
        persona_details=generated_persona,
        age=current_age # Pass age, although birthdate from persona is primary
    )

    if not life_data:
        logger.error("Life summary generation failed critically.")
        console.print(Panel("[bold red]Error:[/bold red] Life summary generation failed.", title="Generation Failed", border_style="red"))
        return None

    # --- Save Results (using determined birthdate) ---
    final_birthdate_str = life_data["persona_details"].get("Birthdate", "UnknownDate")
    persona_name_safe = re.sub(r'[^\w\-]+', '_', generated_persona.get('Name', 'Unknown'))
    output_filename = f"life_summary_{persona_name_safe}_{final_birthdate_str}.json"

    console.print(Rule(f"Saving results to [green]{output_filename}[/green]", style="bold green"))
    try:
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(life_data, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Results saved successfully.")
        console.print(f"[bold green]✔ Results saved successfully to {output_filename}[/bold green]")
    except Exception as e:
        logger.error(f"Error saving results to {output_filename}: {e}", exc_info=True)
        console.print(Panel(f"[bold red]Error saving results:[/bold red] {e}", title="Save Failed", border_style="red"))

    return life_data