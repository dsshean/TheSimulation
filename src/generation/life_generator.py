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
- Generate a plausible birthdate (YYYY-MM-DD) consistent with the generated age and the world type (e.g., future year for SciFi, past year for Fantasy/Historical).

**Required Fields:** Name, Age (integer), Occupation, Current_location (City, State/Country appropriate for the world), Personality_Traits (list, 3-6 adjectives), Birthplace (City, State/Country appropriate for the world), Birthdate (YYYY-MM-DD), Education, Physical_Appearance (brief description), Hobbies, Skills, Languages, Health_Status, Family_Background, Life_Goals, Notable_Achievements, Fears, Strengths, Weaknesses, Ethnicity, Religion (optional), Political_Views (optional), Favorite_Foods, Favorite_Music, Favorite_Books, Favorite_Movies, Pet_Peeves, Dreams, Past_Traumas (optional).

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
Establish a plausible immediate family structure: parents and any siblings. Include brief details consistent with the persona's background and world.
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
    birth_year: int, last_year_to_generate: int, current_year: int, ddgs_instance: DDGS,
    allow_real_context: bool, # <<< Add flag
    world_type: str, world_description: str # <<< Add world context
) -> Optional[Tuple[Dict[int, Tuple[str, Optional[str], Optional[str]]], int, int]]:
    """Generates yearly summaries with location, birthday, and news searches."""
    logger.info(f"Generating yearly summaries with LOCATION JSON ({birth_year}-{last_year_to_generate})...")
    summaries_dict: Dict[int, Tuple[str, Optional[str], Optional[str]]] = {}
    processed_years = set()
    birth_month, birth_day = None, None

    news_context_by_year = {}
    context_instruction = "" # Instruction for LLM based on context flag

    # Only generate summaries if the range is valid (start <= end)
    if birth_year > last_year_to_generate:
        logger.warning(f"Birth year {birth_year} is after the target end year {last_year_to_generate}. Skipping yearly summary generation.")
        return {}, None, None # Return empty dict and None for birth month/day

    for year in range(birth_year, last_year_to_generate + 1):
        search_results_summary = "No external context used." # Default if not allowed/fails
        if allow_real_context:
            context_instruction = "Use the provided News Context and your internal knowledge of real-world events for this year."
            # Perform a news search for the year only if allowed
            time.sleep(1) # Keep rate limiting
            search_query = f"major world events {year}"
            logger.info(f"Performing DDGS search for year {year}: '{search_query}' (Real context allowed)")
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
                search_results_summary = "Use your internal knowledge of this years world and local events." # Keep fallback
                console.print(Panel(
                    "Search failed due to an error.",
                    title=f"News Search Results for {year}",
                    border_style="red",
                    expand=True
                ))
        else:
            # Real context not allowed
            context_instruction = f"Invent plausible fictional events or details for this year, consistent with the world type '{world_type}' and description '{world_description[:100]}...'. Do NOT use real-world events."
            logger.info(f"Skipping DDGS search for year {year} (Real context not allowed)")

        # Store the news context for the year
        news_context_by_year[year] = search_results_summary # Store result or default message

    # Generate the yearly summary
    news_context_combined = "\n".join([f"{year}: {news}" for year, news in news_context_by_year.items()])
    prompt = f"""
    Persona details: {persona_details_str}
    Initial family: {initial_relationships_str}
    Born in {birth_year}. Summaries needed up to end of {last_year_to_generate}.
    World Type: {world_type}
    World Description: {world_description}
    News/External Context (or 'No external context used.'):
    {news_context_combined}

    **Instructions:**
    - Provide a summary of major events for each year from {birth_year} to {last_year_to_generate}.
    - Include the persona's primary location for that year and key life events (personal, professional, relationships).
    - {context_instruction} # <<< Inject conditional instruction here
    - If generating birth year summary, include plausible birth month and day.

    **Output Format:** Respond ONLY with JSON: {{"birth_month": int, "birth_day": int, "summaries": [{{"year": int, "location": str, "summary": str}}]}}
    """
    validated_data = await _call_llm_and_get_validated_data(
        llm_service, prompt, YearlySummariesResponse, f"yearly summary for {birth_year}-{last_year_to_generate}"
    )
    if validated_data:
        try:
            # Extract birth month and day if available
            bm_returned = validated_data.get("birth_month")
            bd_returned = validated_data.get("birth_day")
            if bm_returned and bd_returned:
                birth_month = bm_returned
                birth_day = bd_returned
                logger.info(f"Established Birthday from LLM: {birth_year}-{birth_month:02d}-{birth_day:02d}")

            # Process each summary in the "summaries" list
            for summary_item in validated_data.get("summaries", []):
                year_num = summary_item["year"]
                summary_text = summary_item["summary"]
                location_text = summary_item["location"]

                if year_num in processed_years:
                    logger.warning(f"Duplicate year {year_num}. Skipping.")
                    continue
                # Removed the check: if not (birth_year <= year_num <= last_year_to_generate):

                # Use the news context fetched (or default) for this specific year
                news_context_for_this_year = news_context_by_year.get(year_num, "No external context used.")
                summaries_dict[year_num] = (summary_text, location_text, news_context_for_this_year)
                processed_years.add(year_num)
        except KeyError as e:
            logger.error(f"Missing key in validated data for yearly summaries: {e}")
        except Exception as e:
             logger.error(f"Error processing yearly summary data: {e}", exc_info=True)
    else:
        logger.error(f"Failed to generate yearly summaries for {birth_year}-{last_year_to_generate}.")

    if not summaries_dict:
        logger.warning(f"No valid yearly summaries processed.")
        # Return empty dict but potentially valid birth month/day if found
        return {}, birth_month, birth_day

    logger.info(f"Processed {len(summaries_dict)} yearly summaries with location and news.")
    return summaries_dict, birth_month, birth_day # Return potentially updated birth month/day

async def generate_monthly_summaries_for_year(
    llm_service: LLMService, persona_details_str: str, year: int, yearly_summary: str,
    location_for_year: Optional[str],
    simulated_current_year: int, simulated_current_month: int, # <<< Use simulated date
    ddgs_instance: DDGS,
    allow_real_context: bool, # <<< Add flag
    world_type: str, world_description: str # <<< Add world context
) -> Optional[Dict[int, Tuple[str, Optional[str], Optional[str]]]]:
    """Generates monthly summaries with location and includes news context."""
    logger.info(f"Generating monthly summaries for {year} (Loc: {location_for_year})...")
    monthly_news_context_by_month = {}
    context_instruction = "" # Instruction for LLM based on context flag

    # Determine months to search/generate based on simulated time
    months_to_search = []
    months_to_generate_up_to = 12 # Default: all months for past years
    is_current_sim_year = (year == simulated_current_year)

    if is_current_sim_year:
        months_to_generate_up_to = simulated_current_month # Generate up to simulated current month
        # Search context only for the most recent simulated months if allowed
        current_month_key = f"{year}-{simulated_current_month:02d}"
        months_to_search.append(current_month_key)
        if simulated_current_month == 1:
            prev_month_key = f"{year-1}-12"
            # Check if previous year is within persona's lifetime (needs birth_year if strict check desired)
            months_to_search.append(prev_month_key)
        else:
            prev_month_key = f"{year}-{simulated_current_month-1:02d}"
            months_to_search.append(prev_month_key)
        date_string_for_prompt = f"months 1 through {months_to_generate_up_to}"
        console.print(f"Targeting months for {year}: {date_string_for_prompt} (News search: {months_to_search if allow_real_context else 'None'})")
    else: # Past year
        date_string_for_prompt = f"all months" # Adjust prompt instruction
        logger.info(f"Generating all months for past year {year}. Skipping monthly news search.")
        allow_real_context = False # Force disable news search for past years

    # Perform a news search for each targeted month if allowed
    for month_key in months_to_search: # Only iterate targeted months for search
        search_results_summary = "No external context used." # Default
        if allow_real_context: # Check flag AND if it's a month we target
            context_instruction = "Use the provided News Context and your internal knowledge of real-world events for these months."
            search_query = f"major world events news {location_for_year or 'the world'} {month_key}"
            logger.info(f"Performing DDGS search for month {month_key}: '{search_query}' (Real context allowed)")
            time.sleep(1)
            try:
                search_results_list: List[Dict] = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=5)
                if search_results_list:
                    formatted_results = [f"- {r.get('title', '')}: {r.get('body', '')[:100]}..." for r in search_results_list]
                    search_results_summary = "\n".join(formatted_results)
                    console.print(Panel("\n".join(formatted_results), title=f"News Search Results for {month_key}", border_style="blue", expand=True))
                else:
                    search_results_summary = "Search returned no results."
                    console.print(Panel("No results found.", title=f"News Search Results for {month_key}", border_style="red", expand=True))
                logger.debug(f"DDGS monthly context summary for {month_key}:\n{search_results_summary}")
            except Exception as search_err:
                logger.error(f"DDGS search failed for month {month_key}: {search_err}")
                search_results_summary = "Search failed." # Keep fallback
                console.print(Panel("Search failed due to an error.", title=f"News Search Results for {month_key}", border_style="red", expand=True))
        else:
            # Real context not allowed or not a targeted month
            context_instruction = f"Invent plausible fictional events or details for these months, consistent with the world type '{world_type}' and description '{world_description[:100]}...'. Do NOT use real-world events."
            if month_key in months_to_search: # Only log skip if it was targeted
                logger.info(f"Skipping DDGS search for month {month_key} (Real context not allowed or past year)")

        # Store the news context for the month (even if default)
        monthly_news_context_by_month[month_key] = search_results_summary

    # Generate the monthly summaries
    news_context_combined = "\n".join([f"{month_key}: {news}" for month_key, news in monthly_news_context_by_month.items() if month_key in months_to_search])
    # Adjust prompt based on whether it's current year or past year
    if is_current_sim_year:
        task_instruction = f"Create summaries for **MONTHS 1 through {months_to_generate_up_to}** of the current year {year}."
    else: # Past year
        task_instruction = f"Create summaries for **ALL MONTHS** of the past year {year}."

    prompt = f"""
Persona: {persona_details_str}, Year: {year}, Location: {location_for_year or 'inferred'}, Yearly Context: "{yearly_summary}"
World Type: {world_type}
World Description: {world_description}
News Search Context (if applicable): {news_context_combined or 'None'}

**Task:** {task_instruction}
**Instructions:**
- Generate a plausible summary for each required month.
- Include the persona's location (can usually be inferred from the year's location).
- {context_instruction} # <<< Inject conditional instruction here

**Output Format:** Respond ONLY with JSON: {{"summaries": [{{"month": int, "location": str, "summary": str}}]}}
"""
    validated_data = await _call_llm_and_get_validated_data(llm_service, prompt, MonthlySummariesResponse, f"monthly summaries year {year}")
    if validated_data:
        summaries_list = validated_data.get("summaries", []) # Use .get for safety
        summaries_dict: Dict[int, Tuple[str, Optional[str], Optional[str]]] = {}
        processed_months = set()
        for item_dict in summaries_list:
            try:
                month_num = item_dict["month"]
                summary_text = item_dict["summary"]
                location_text = item_dict["location"]
            except KeyError as e:
                logger.warning(f"Skipping monthly summary item due to missing key: {e}. Item: {item_dict}")
                continue

            if not (1 <= month_num <= months_to_generate_up_to): # Check against calculated limit
                logger.warning(f"Invalid month {month_num} in response for year {year}. Skipping.")
                continue
            if month_num in processed_months:
                logger.warning(f"Duplicate month {month_num} in response for year {year}. Skipping.")
                continue

            # Convert the month number to the "YYYY-MM" format to match the keys in monthly_news_context_by_month
            month_key = f"{year}-{month_num:02d}"
            # Get news context if it exists, otherwise use default
            news_context = monthly_news_context_by_month.get(month_key, "No external context used.")

            summaries_dict[month_num] = (summary_text, location_text, news_context)
            processed_months.add(month_num)
        logger.info(f"Processed {len(summaries_dict)} monthly summaries for {year}.")
        return summaries_dict
    else:
        logger.error(f"Failed validated monthly summaries for {year}.")
        return None

async def generate_daily_summaries_for_month(
    llm_service: LLMService, persona_details_str: str, year: int, month: int, monthly_summary: str,
    yearly_summary: str, location_for_month: Optional[str],
    simulated_current_year: int, simulated_current_month: int, simulated_current_day: int, # <<< Use simulated date
    ddgs_instance: DDGS,
    allow_real_context: bool, # <<< Add flag
    world_type: str, world_description: str # <<< Add world context
) -> Optional[Dict[int, Tuple[str, Optional[str], Optional[str]]]]: # <<< Return type changed
    """Generates daily summaries for the last 7 days with location and includes news context."""
    logger.info(f"Generating daily summaries for {year}-{month:02d} (Loc: {location_for_month})...")
    daily_news_context_by_day = {}
    context_instruction = "" # Instruction for LLM based on context flag

    # Determine the range of days to generate
    simulated_current_date_for_month = date(simulated_current_year, simulated_current_month, simulated_current_day)
    is_current_sim_month = (year == simulated_current_year and month == simulated_current_month)

    if is_current_sim_month:
        # Generate for the last 7 days ending on the simulated current day within this month
        days_to_generate_dates = [(simulated_current_date_for_month - timedelta(days=i)) for i in range(7)]
        days_to_generate_dates = [d for d in days_to_generate_dates if d.year == year and d.month == month] # Filter for target month
    else: # Past month/year - generate for all days
        try:
            days_in_month_count = calendar.monthrange(year, month)[1]
            days_to_generate_dates = [date(year, month, d) for d in range(1, days_in_month_count + 1)]
        except ValueError:
             logger.error(f"Invalid year/month combination: {year}-{month}. Cannot generate daily summaries.")
             return None

    if not days_to_generate_dates:
        logger.warning(f"No valid days found to generate for the target month {year}-{month:02d}.")
        return None # Return None if no days to process

    # Perform a news search for each relevant day if allowed and it's the current simulated month
    search_days = days_to_generate_dates if is_current_sim_month else [] # Only search recent days

    for day_date in search_days: # Only iterate days needing search
        day_key = day_date.strftime("%Y-%m-%d")
        day_num = day_date.day
        search_results_summary = "No external context used." # Default
        if allow_real_context:
            context_instruction = "Use the provided News Context and your internal knowledge of real-world events for these days."
            time.sleep(1)
            search_query = f"events news {location_for_month or 'region'} {day_key}"
            logger.info(f"Performing DDGS search for day {day_key}: '{search_query}' (Real context allowed)")
            try:
                search_results_list: List[Dict] = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=5)
                if search_results_list:
                    formatted_results = [f"- {r.get('title', '')}: {r.get('body', '')[:100]}..." for r in search_results_list]
                    search_results_summary = "\n".join(formatted_results)
                    console.print(Panel("\n".join(formatted_results), title=f"News Search Results for {day_key}", border_style="blue", expand=True))
                else:
                    search_results_summary = "Search returned no results."
                    console.print(Panel("No results found.", title=f"News Search Results for {day_key}", border_style="red", expand=True))
                logger.debug(f"DDGS daily context summary for {day_key}:\n{search_results_summary}")
            except Exception as search_err:
                logger.error(f"DDGS search failed for day {day_key}: {search_err}")
                search_results_summary = "Search failed." # Keep fallback
                console.print(Panel("Search failed due to an error.", title=f"News Search Results for {day_key}", border_style="red", expand=True))
        else:
            # Real context not allowed
            context_instruction = f"Invent plausible fictional events or details for these days, consistent with the world type '{world_type}' and description '{world_description[:100]}...'. Do NOT use real-world events."
            logger.info(f"Skipping DDGS search for day {day_key} (Real context not allowed)")

        # Store the news context for the day number
        daily_news_context_by_day[day_num] = search_results_summary # Store result or default message

    # Combine all daily news contexts into one string for the prompt
    news_context_combined = "\n".join([f"{day_num}: {news}" for day_num, news in daily_news_context_by_day.items()]) # Use day_num as key
    days_to_generate_nums = sorted([d.day for d in days_to_generate_dates]) # Use the calculated dates

    # Adjust prompt based on whether it's the current simulated month or a past one
    if is_current_sim_month:
        task_instruction = f"Generate summaries for the following **RECENT DAYS ONLY**: {', '.join(map(str, days_to_generate_nums))} of {calendar.month_name[month]}, {year}."
    else: # Past month
        task_instruction = f"Generate summaries for **ALL DAYS** ({', '.join(map(str, days_to_generate_nums))}) of {calendar.month_name[month]}, {year}."

    # Generate the daily summaries
    prompt = f"""
Persona: {persona_details_str}, Context: Year={year}("{yearly_summary}"), Month={month}("{monthly_summary}"), Location: {location_for_month or 'inferred'}
World Type: {world_type}
World Description: {world_description}
News Context (if applicable): {news_context_combined or 'None'}

**Task:** {task_instruction}
**Instructions:**
- Generate a plausible summary for each required day.
- Include the persona's location (can usually be inferred from the month's location).
- {context_instruction} # <<< Inject conditional instruction here

**Output Format:** Respond ONLY with JSON: {{"summaries": [{{"day": int, "location": str, "summary": str}}]}}
"""
    validated_data = await _call_llm_and_get_validated_data(llm_service, prompt, DailySummariesResponse, f"daily summaries {year}-{month:02d}")
    if validated_data:
        summaries_list = validated_data.get("summaries", []) # Use .get for safety
        summaries_dict: Dict[int, Tuple[str, Optional[str], Optional[str]]] = {}
        processed_days = set()
        for item_dict in summaries_list:
            try:
                day_num = item_dict["day"]
                summary_text = item_dict["summary"]
                location_text = item_dict["location"]
            except KeyError as e:
                logger.warning(f"Skipping daily summary item due to missing key: {e}. Item: {item_dict}")
                continue

            # Validate day number against the days we actually requested
            if day_num not in days_to_generate_nums:
                logger.warning(f"LLM returned summary for day {day_num}, which was not requested ({days_to_generate_nums}). Skipping.")
                continue
            # Basic day validity check
            try:
                days_in_month = calendar.monthrange(year, month)[1]
                if not (1 <= day_num <= days_in_month):
                    logger.warning(f"Invalid day {day_num} for month {month}. Skipping.")
                    continue
            except ValueError: # Handle invalid month
                 logger.warning(f"Invalid month {month} when checking day {day_num}. Skipping.")
                 continue

            if day_num in processed_days:
                logger.warning(f"Duplicate day {day_num} in response for {year}-{month:02d}. Skipping.")
                continue

            # Get news context for this day number
            news_context = daily_news_context_by_day.get(day_num, "No external context used.")
            summaries_dict[day_num] = (summary_text, location_text, news_context)
            processed_days.add(day_num)

        logger.info(f"Processed {len(summaries_dict)} daily summaries for {year}-{month:02d}.")
        return summaries_dict # Return only the dictionary
    else:
        logger.error(f"Failed validated daily summaries for {year}-{month:02d}.")
        return None

async def generate_hourly_breakdown_for_day(
    llm_service: LLMService, persona_details_str: str, year: int, month: int, day: int, daily_summary: str,
    monthly_summary: str, yearly_summary: str, location_for_day: Optional[str],
    simulated_current_year: int, simulated_current_month: int, simulated_current_day: int, simulated_current_hour: int, # <<< Use simulated date/time
    ddgs_instance: DDGS,
    allow_real_context: bool, # <<< Add flag
    world_type: str, world_description: str # <<< Add world context
) -> Optional[Dict[str, Any]]: # <<< Return type changed
    """Generates hourly breakdown with location and includes news searches."""
    logger.info(f"Generating hourly breakdown for {year}-{month:02d}-{day:02d} (Loc: {location_for_day})...")
    search_results_summary = "No external context used." # Default
    day_key = f"{year}-{month:02d}-{day:02d}"
    context_instruction = "" # Instruction for LLM based on context flag
    is_current_simulated_day = (year == simulated_current_year and month == simulated_current_month and day == simulated_current_day)

    if allow_real_context and is_current_simulated_day: # Only search if allowed AND it's the current simulated day
        context_instruction = "Use the provided News Context and your internal knowledge of real-world events for this day."
        search_query = f"events news {location_for_day or 'region'} {day_key}"
        logger.info(f"Performing DDGS search: '{search_query}' (Real context allowed)")
        try:
            search_results_list: List[Dict] = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=5)
            if search_results_list:
                formatted_results = [f"- {r.get('title', '')}: {r.get('body', '')[:100]}..." for r in search_results_list]
                search_results_summary = "\n".join(formatted_results)
                console.print(Panel("\n".join(formatted_results), title=f"News Search Results for {day_key}", border_style="blue", expand=True))
            else:
                search_results_summary = "Search returned no results."
                console.print(Panel("No results found.", title=f"News Search Results for {day_key}", border_style="red", expand=True))
            logger.debug(f"DDGS daily context summary:\n{search_results_summary}")
        except Exception as search_err:
            logger.error(f"DDGS search failed: {search_err}")
            search_results_summary = "Search failed." # Keep fallback
            console.print(Panel("Search failed due to an error.", title=f"News Search Results for {day_key}", border_style="red", expand=True))
    else:
        # Real context not allowed OR it's a past day
        context_instruction = f"Invent plausible fictional activities and details for the hours of this day, consistent with the world type '{world_type}' and description '{world_description[:100]}...'. Do NOT use real-world events."
        if is_current_simulated_day: # Log skip only if it was the current day but context wasn't allowed
             logger.info(f"Skipping DDGS search for {day_key} (Real context not allowed)")
        # For past days, no search is attempted anyway

    # Determine hours to generate
    hours_to_generate_up_to = simulated_current_hour if is_current_simulated_day else 23
    task_instruction = f"Generate hour-by-hour breakdown (0-{hours_to_generate_up_to})." if is_current_simulated_day else "Generate hour-by-hour breakdown (0-23)."

    # Generate the hourly breakdown
    prompt = f"""
Persona: {persona_details_str}, Date: {day_key}, Location: {location_for_day or 'inferred'}
Context: Year="{yearly_summary}", Month="{monthly_summary}", Day="{daily_summary}"
World Type: {world_type}
World Description: {world_description}
News/External Context (if applicable): "{search_results_summary}"

**Task:** {task_instruction}
**Instructions:**
- Generate a plausible activity and location for each required hour (0 through {hours_to_generate_up_to}).
- {context_instruction} # <<< Inject conditional instruction here

**Output Format:** Respond ONLY with JSON: {{"activities": [{{"hour": int, "location": str, "activity": str}}]}}
"""
    validated_data = await _call_llm_and_get_validated_data(llm_service, prompt, HourlyBreakdownResponse, f"hourly breakdown {day_key}")
    if validated_data:
        activities_list = validated_data.get("activities", []) # Use .get for safety
        activities_dict: Dict[int, Tuple[str, Optional[str]]] = {}
        processed_hours = set()
        for item_dict in activities_list:
            try:
                hour_num = item_dict["hour"]
                activity_text = item_dict["activity"]
                location_text = item_dict["location"]
            except KeyError as e:
                 logger.warning(f"Skipping hourly activity item due to missing key: {e}. Item: {item_dict}")
                 continue

            if not (0 <= hour_num <= hours_to_generate_up_to): # Check against calculated limit
                logger.warning(f"Invalid hour {hour_num} in response for {day_key}. Skipping.")
                continue
            if hour_num in processed_hours:
                logger.warning(f"Duplicate hour {hour_num} in response for {day_key}. Skipping.")
                continue
            activities_dict[hour_num] = (activity_text, location_text)
            processed_hours.add(hour_num)

        if not activities_dict:
            logger.warning(f"No valid hourly activities generated for {day_key}.")
            # Return structure with empty activities but include news
            return {"activities": {}, "news": search_results_summary}

        logger.info(f"Processed {len(activities_dict)} hourly activities for {day_key}.")
        return {"activities": activities_dict, "news": search_results_summary}
    else:
        logger.error(f"Failed validated hourly activities for {day_key}.")
        return None

# <<< IMPORTANT: Define generate_life_summary_sequentially BEFORE generate_new_simulacra_background >>>
async def generate_life_summary_sequentially(
    llm_service: LLMService,
    persona_details: Dict[str, Any],
    # age: int, # Age is now derived from birthdate and generation time
    generation_timestamp: datetime, # Pass the timestamp used for "now"
    allow_real_context: bool, # <<< Add flag
    world_type: str, world_description: str # <<< Add world context
) -> Optional[Dict[str, Any]]:
    """
    Orchestrates generation with simplified loop logic, calling the original generator functions.
    Updates the summary_tree only after life_summary is fully generated.
    """
    # --- Setup: Date/Time and Persona Info ---
    # Use the passed generation_timestamp as the reference "current" time
    simulated_current_dt = generation_timestamp
    simulated_current_date = simulated_current_dt.date()
    simulated_current_year = simulated_current_dt.year
    simulated_current_month = simulated_current_dt.month
    simulated_current_day = simulated_current_dt.day
    simulated_current_hour = simulated_current_dt.hour # Use this for pruning hourly
    console.print(f"Generation Reference Time ('Now'): [cyan]{simulated_current_dt.strftime('%Y-%m-%d %H:%M:%S %Z')}[/cyan]")

    # --- Birthdate Handling ---
    birthdate_str = persona_details.get("Birthdate")
    birthdate = None
    if birthdate_str:
        try:
            birthdate = datetime.strptime(str(birthdate_str), "%Y-%m-%d").date() # Ensure str
        except (ValueError, TypeError):
            logger.warning(f"Invalid Birthdate format '{birthdate_str}'. Estimating.")
            birthdate_str = None # Clear invalid string to trigger estimation

    if not birthdate_str:
        # Estimate if missing or invalid
        # Use a default age if 'Age' is also missing from persona
        age_for_estimation = persona_details.get("Age", 30)
        birth_year_approx = simulated_current_year - age_for_estimation
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
    # Calculate age based on simulated current date
    actual_age = simulated_current_year - birth_year - ((simulated_current_month, simulated_current_day) < (birth_month, birth_day))
    logger.info(f"Persona: {persona_details.get('Name', 'N/A')}, Birthdate: {birthdate}, Actual Age: {actual_age}")
    persona_details["Age"] = actual_age # Ensure Age field matches calculated age

    # Note: actual_age can be negative if born in the future, but we still proceed.
    # The end_year_for_generation will correctly reflect their current age relative to birth.

    end_year_for_generation = birth_year + actual_age
    logger.info(f"Generation will proceed up to year: {end_year_for_generation} (Birth: {birth_year}, Age: {actual_age})")

    # --- Initialize main data structure ---
    life_summary = {
        "persona_details": persona_details,
        "generation_info": { # Reflects the simulated "now"
            "generated_at": generation_timestamp.isoformat(), # Timestamp when generation ran
            "current_year": simulated_current_year,
            "current_month": simulated_current_month,
            "current_day": simulated_current_day,
            "current_hour": simulated_current_hour,
        },
        "birth_year": birth_year,
        "birth_month": birth_month, # Store initial/estimated
        "birth_day": birth_day,   # Store initial/estimated
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
            birth_year, end_year_for_generation, # <<< Use calculated end year
            simulated_current_year, # Still pass simulated current year for context if needed inside
            ddgs_instance=ddgs_search_tool,
            allow_real_context=allow_real_context, # <<< Pass flag
            world_type=world_type, world_description=world_description # <<< Pass world context
        )
        if yearly_result:
            yearly_summaries_tuples, bm_returned, bd_returned = yearly_result
            # Update birth month/day if LLM provided them AND they weren't set initially
            if bm_returned and bd_returned:
                 if not life_summary.get("birth_month") or not life_summary.get("birth_day"):
                     life_summary["birth_month"] = bm_returned
                     life_summary["birth_day"] = bd_returned
                     logger.info(f"Updated birth month/day from yearly summary: {bm_returned}/{bd_returned}")
                 elif life_summary["birth_month"] != bm_returned or life_summary["birth_day"] != bd_returned:
                      logger.warning(f"LLM yearly summary suggested different birth month/day ({bm_returned}/{bd_returned}) than initial ({life_summary['birth_month']}/{life_summary['birth_day']}). Keeping initial.")

            for year, (summary, location, news) in yearly_summaries_tuples.items():
                life_summary["yearly_summaries"][year] = {
                    "summary": summary, "location": location, "news": news
                }
    except Exception as e:
        logger.error(f"Error generating yearly summaries: {e}", exc_info=True)

    # --- Step 2: Generate Monthly Summaries ---
    console.print(Rule("Generating Monthly Summaries (Relevant Months)", style="bold yellow"))
    try:
        # Determine the months to generate (current and previous relative to simulated date)
        months_to_generate = []
        current_sim_date_obj = date(simulated_current_year, simulated_current_month, 1) # Use object for comparison
        months_to_generate.append((current_sim_date_obj.year, current_sim_date_obj.month))
        # Previous month calculation
        if current_sim_date_obj.month == 1:
            prev_month_date = date(current_sim_date_obj.year - 1, 12, 1)
        else:
            prev_month_date = date(current_sim_date_obj.year, current_sim_date_obj.month - 1, 1)
        # Only generate previous month if it's on or after birthdate's year
        if prev_month_date >= date(birth_year, 1, 1): # Compare dates
             months_to_generate.append((prev_month_date.year, prev_month_date.month))

        for year, month in months_to_generate:
            # Ensure we don't try to generate for months before birth month in birth year
            if year == birth_year and month < birth_month:
                continue
            # Ensure we don't try to generate for months after the character's current point
            if date(year, month, 1) > date(end_year_for_generation, birth_month, 1): # Approx check
                continue

            yearly_context = life_summary["yearly_summaries"].get(year, {})
            monthly_result = await generate_monthly_summaries_for_year(
                llm_service, persona_details_str, year,
                yearly_context.get("summary", "No yearly summary available."),
                yearly_context.get("location"),
                simulated_current_year, simulated_current_month, # Pass simulated date
                ddgs_instance=ddgs_search_tool,
                allow_real_context=allow_real_context, # <<< Pass flag
                world_type=world_type, world_description=world_description # <<< Pass world context
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
    console.print(Rule("Generating Daily Summaries (Last 7 Simulated Days)", style="bold yellow"))
    try:
        # <<< FIX: Calculate 'character's current date' based on end_year and birth month/day >>>
        # Use birth month/day but clamp day if invalid for that month in the end year
        try:
            days_in_end_month = calendar.monthrange(end_year_for_generation, birth_month)[1]
            clamped_birth_day = min(birth_day, days_in_end_month)
            character_current_date = date(end_year_for_generation, birth_month, clamped_birth_day)
        except ValueError: # Handle potential invalid month/year combo if logic allows edge cases
            logger.warning(f"Could not form valid date for character's current point: {end_year_for_generation}-{birth_month}-{birth_day}. Using end year approx.")
            character_current_date = date(end_year_for_generation, 1, 1) # Fallback

        days_to_generate_dates = [(character_current_date - timedelta(days=i)) for i in range(7)] # Use character's current date
        # Filter days to be on or after birthdate
        days_to_generate_dates = [d for d in days_to_generate_dates if d >= birthdate]

        for day_date in days_to_generate_dates:
            year, month, day_num = day_date.year, day_date.month, day_date.day
            monthly_context = life_summary["monthly_summaries"].get(year, {}).get(month, {})
            yearly_context_summary = life_summary["yearly_summaries"].get(year, {}).get("summary", "No yearly summary.")
            daily_result_dict = await generate_daily_summaries_for_month( # Renamed variable
                llm_service, persona_details_str, year, month,
                monthly_context.get("summary", "No monthly summary available."),
                yearly_context_summary,
                monthly_context.get("location"), # Use month's location as context
                simulated_current_year, simulated_current_month, simulated_current_day, # Pass simulated date
                ddgs_instance=ddgs_search_tool,
                allow_real_context=allow_real_context, # <<< Pass flag
                world_type=world_type, world_description=world_description # <<< Pass world context
            )
            if daily_result_dict:
                 life_summary["daily_summaries"].setdefault(year, {}).setdefault(month, {}).update({
                    d: {"summary": summary, "location": location, "news": news}
                    for d, (summary, location, news) in daily_result_dict.items()
                    # Only add if the day 'd' corresponds to one of the dates we generated for
                    if date(year, month, d) in days_to_generate_dates
                })
    except Exception as e:
        logger.error(f"Error generating daily summaries: {e}", exc_info=True)

    # --- Step 4: Generate Hourly Breakdowns ---
    console.print(Rule("Generating Hourly Breakdowns (Simulated Today & Yesterday)", style="bold yellow"))
    try:
        # Use character_current_date calculated in Step 3
        days_for_hourly_dates = [character_current_date, character_current_date - timedelta(days=1)] # Use character's current date
        # Filter days to be on or after birthdate
        days_for_hourly_dates = [d for d in days_for_hourly_dates if d >= birthdate]

        for day_date in days_for_hourly_dates:
            year, month, day_num = day_date.year, day_date.month, day_date.day
            daily_context = life_summary["daily_summaries"].get(year, {}).get(month, {}).get(day_num, {})
            monthly_context_summary = life_summary["monthly_summaries"].get(year, {}).get(month, {}).get("summary", "No monthly summary.")
            yearly_context_summary = life_summary["yearly_summaries"].get(year, {}).get("summary", "No yearly summary.")
            hourly_result = await generate_hourly_breakdown_for_day(
                llm_service, persona_details_str, year, month, day_num,
                daily_context.get("summary", "No daily summary available."),
                monthly_context_summary,
                yearly_context_summary,
                daily_context.get("location"), # Use day's location as context
                # Pass simulated date/time
                simulated_current_year, simulated_current_month, simulated_current_day, simulated_current_hour,
                ddgs_instance=ddgs_search_tool,
                allow_real_context=allow_real_context, # <<< Pass flag
                world_type=world_type, world_description=world_description # <<< Pass world context
            )
            if hourly_result:
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
    birth_info_node.add(f"Year: {life_summary.get('birth_year')}")
    birth_info_node.add(f"Month: {life_summary.get('birth_month')}")
    birth_info_node.add(f"Day: {life_summary.get('birth_day')}")
    # --- Relationships ---
    relationships_node = summary_tree.add(f"[bold green]Initial Relationships[/bold green]")
    if life_summary.get("initial_relationships"):
        parents = life_summary["initial_relationships"].get("parents", [])
        siblings = life_summary["initial_relationships"].get("siblings", [])
        for parent in parents:
            relationships_node.add(f"Parent: {parent.get('name', 'Unknown')}")
        for sibling in siblings:
            relationships_node.add(f"Sibling: {sibling.get('name', 'Unknown')}")
    # --- Yearly Summaries ---
    yearly_summaries_node = summary_tree.add(f"[bold green]Yearly Summaries[/bold green]")
    for year, data in sorted(life_summary.get("yearly_summaries", {}).items()): # Sort by year
        yearly_summaries_node.add(f"[bold yellow]{year}[/bold yellow]: {data.get('summary', 'No summary')}")
    # --- Monthly Summaries ---
    monthly_summaries_node = summary_tree.add(f"[bold green]Monthly Summaries[/bold green]")
    for year, months in sorted(life_summary.get("monthly_summaries", {}).items()): # Sort by year
        year_node = monthly_summaries_node.add(f"[bold yellow]{year}[/bold yellow]")
        for month, data in sorted(months.items()): # Sort by month
            year_node.add(f"[bold yellow]{month}[/bold yellow]: {data.get('summary', 'No summary')}")
    # --- Daily Summaries ---
    daily_summaries_node = summary_tree.add(f"[bold green]Daily Summaries[/bold green]")
    for year, months in sorted(life_summary.get("daily_summaries", {}).items()): # Sort by year
        year_node = daily_summaries_node.add(f"[bold yellow]{year}[/bold yellow]")
        for month, days in sorted(months.items()): # Sort by month
            month_node = year_node.add(f"[bold yellow]{month}[/bold yellow]")
            for day, data in sorted(days.items()): # Sort by day
                month_node.add(f"[bold yellow]{day}[/bold yellow]: {data.get('summary', 'No summary')}")
    # --- Hourly Breakdowns ---
    hourly_breakdowns_node = summary_tree.add(f"[bold green]Hourly Breakdowns[/bold green]")
    for year, months in sorted(life_summary.get("hourly_breakdowns", {}).items()): # Sort by year
        year_node = hourly_breakdowns_node.add(f"[bold yellow]{year}[/bold yellow]")
        for month, days in sorted(months.items()): # Sort by month
            month_node = year_node.add(f"[bold yellow]{month}[/bold yellow]")
            for day, data in sorted(days.items()): # Sort by day
                day_node = month_node.add(f"[bold yellow]{day}[/bold yellow]")
                for hour, activity_tuple in sorted(data.get("activities", {}).items()): # Sort by hour
                    # Ensure activity_tuple is treated as a tuple
                    activity_text = activity_tuple[0] if isinstance(activity_tuple, tuple) and len(activity_tuple) > 0 else "Unknown Activity"
                    location_text = activity_tuple[1] if isinstance(activity_tuple, tuple) and len(activity_tuple) > 1 else "Unknown Location"
                    day_node.add(f"[bold yellow]{hour:02d}:00[/bold yellow] - {activity_text} at {location_text}") # Format hour
    console.print(Rule("Generation Complete", style="bold green"))
    console.print(summary_tree)

    return life_summary

async def generate_new_simulacra_background(
    sim_id: str,
    world_instance_uuid: Optional[str],
    world_type: str,
    world_description: str,
    allow_real_context: bool, # <<< Add flag
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
    logger.info(f"Allow Real Context (News Searches): {allow_real_context}") # Log the flag

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
    )

    # --- Fallback and Birthdate Handling ---
    if not generated_persona:
        logger.error("Could not generate persona via LLM. Using fallback.")
        fallback_age = 30 # Example fallback age
        # Estimate birth year based on real current year for fallback
        fallback_birth_year = datetime.now(timezone.utc).year - fallback_age
        generated_persona = {
            "Name": "Alex Default", "Age": fallback_age, "Occupation": "Archivist",
            "Current_location": "Default City, Default State", "Personality_Traits": ["Methodical", "Calm", "Inquisitive"],
            "Birthplace": "Old Town, Default State", "Education": "Degree in History",
            "Birthdate": f"{fallback_birth_year}-07-15", # Example fixed birthdate
            "Physical_Appearance": "Average height, tidy appearance.",
            # Add other required fields from PersonaDetailsResponse if using fallback
            "Hobbies": "Reading old texts", "Skills": "Organization, Research", "Languages": "Native Language",
            "Health_Status": "Good", "Family_Background": "Middle class", "Life_Goals": "Uncover a hidden truth",
            "Notable_Achievements": "Published a small paper", "Fears": "Losing records", "Strengths": "Detail-oriented",
            "Weaknesses": "Socially awkward", "Ethnicity": "Default", "Religion": None, "Political_Views": None,
            "Favorite_Foods": "Simple meals", "Favorite_Music": "Classical", "Favorite_Books": "Histories",
            "Favorite_Movies": "Documentaries", "Pet_Peeves": "Misfiled documents", "Dreams": "Finding a lost library",
            "Past_Traumas": None
        }
        console.print(Panel(pretty_repr(generated_persona), title="Using Fallback Persona", border_style="red", expand=False))
    else:
         # Ensure birthdate exists and is valid, estimate if missing/invalid
        birthdate_str = generated_persona.get("Birthdate")
        valid_birthdate = False
        if birthdate_str and isinstance(birthdate_str, str) and re.match(r"\d{4}-\d{2}-\d{2}", birthdate_str): # Check type and format
            try:
                # Check if the date is plausible (e.g., not year 9999 for a real world sim)
                temp_date = datetime.strptime(birthdate_str, "%Y-%m-%d").date()
                # Add more sophisticated plausibility checks if needed based on world_type
                valid_birthdate = True
            except ValueError:
                logger.warning(f"Generated persona has invalid date format '{birthdate_str}'. Estimating.")

        if not valid_birthdate:
             logger.warning("Generated persona missing valid 'Birthdate'. Estimating.")
             age_from_persona = generated_persona.get("Age", 30) # Use LLM age if available
             # Estimate birth year based on real current year
             birth_year_est = datetime.now(timezone.utc).year - age_from_persona
             # Consider a more robust default/estimation (e.g., random month/day)
             generated_persona["Birthdate"] = f"{birth_year_est}-07-15" # Ensure it's a string

        console.print(Panel(pretty_repr(generated_persona), title="Generated Persona", border_style="blue", expand=False))

    # --- Determine the correct 'generation_timestamp' based on world type ---
    generation_start_time = None
    try:
        birthdate_obj = datetime.strptime(generated_persona["Birthdate"], "%Y-%m-%d").date()
        # Use the age generated by the LLM or fallback
        current_age = generated_persona.get("Age", 30) # This age is relative to the character's present

        if world_type == "real" and allow_real_context: # Assume realtime if context is allowed
            generation_start_time = datetime.now(timezone.utc)
            logger.info(f"Using real-world current time as generation reference: {generation_start_time.isoformat()}")
        else: # SciFi, Fantasy, Custom, Historical (without specific date)
            # Calculate the character's "current" date based on their age
            target_year = birthdate_obj.year + current_age
            target_month = birthdate_obj.month
            target_day = birthdate_obj.day
            try:
                # Clamp day if necessary (e.g., Feb 29)
                days_in_target_month = calendar.monthrange(target_year, target_month)[1]
                clamped_target_day = min(target_day, days_in_target_month)
                character_present_date = date(target_year, target_month, clamped_target_day)
            except ValueError:
                logger.warning(f"Could not form valid date for character's present: {target_year}-{target_month}-{target_day}. Using year start.")
                character_present_date = date(target_year, 1, 1) # Fallback

            # Create a datetime object, defaulting to noon UTC for non-realtime
            generation_start_time = datetime(character_present_date.year, character_present_date.month, character_present_date.day, 12, 0, 0, tzinfo=timezone.utc)
            logger.info(f"Using character's calculated present ({character_present_date}) as generation reference: {generation_start_time.isoformat()}")
    except Exception as e:
        logger.error(f"Error determining generation timestamp: {e}. Falling back to real-time.", exc_info=True)
        generation_start_time = datetime.now(timezone.utc) # Fallback on error

    # --- Run Life Summary Generation ---
    console.print(Rule(f"Generating Life Summary (Birthdate: {generated_persona.get('Birthdate')})", style="bold yellow"))
    # Pass the current real-world timestamp as the reference "now" for generation
    life_data = await generate_life_summary_sequentially(
        llm_service=llm_service,
        persona_details=generated_persona, # Pass the potentially updated persona
        # age=current_age, # Age is derived inside sequential now
        generation_timestamp=generation_start_time, # Pass timestamp
        allow_real_context=allow_real_context, # <<< Pass flag
        world_type=world_type, world_description=world_description # <<< Pass world context
    )

    if not life_data:
        logger.error("Life summary generation failed critically.")
        console.print(Panel("[bold red]Error:[/bold red] Life summary generation failed.", title="Generation Failed", border_style="red"))
        return None # Return None if the core generation fails

    # --- Add Metadata and Save Results ---
    life_data["sim_id"] = sim_id
    life_data["world_instance_uuid"] = world_instance_uuid
    # Ensure generation_info exists and update generated_at (it's set inside sequential now)
    if "generation_info" in life_data:
        life_data["generation_info"]["generated_at"] = generation_start_time.isoformat() # Use the start time
    else:
        # If generation_info is missing, add it using the start time
        life_data["generation_info"] = {
            "generated_at": generation_start_time.isoformat(),
            "current_year": generation_start_time.year,
            "current_month": generation_start_time.month,
            "current_day": generation_start_time.day,
            "current_hour": generation_start_time.hour,
        }
        logger.warning("life_data was missing 'generation_info', added default structure.")


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

    if "persona_details" not in life_data:
        logger.error("Generated life_data is missing the 'persona_details' key before returning.")

    return life_data
