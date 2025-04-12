import re
import math
import logging
import asyncio
import json
import calendar
import os
from datetime import date, timedelta, datetime
from typing import Dict, Any, Optional, List, Tuple, Type, TypeVar, Union
from pydantic import BaseModel, ValidationError
# Removed Langchain import, using duckduckgo_search directly
from duckduckgo_search import DDGS # Import the main class
# Assume src imports work
from src.llm_service import LLMService
from src.models import ( InitialRelationshipsResponse, Person, YearlySummariesResponse, MonthlySummariesResponse, DailySummariesResponse, HourlyBreakdownResponse, YearSummary, MonthSummary, DaySummary, HourActivity, PersonaDetailsResponse )

# --- Configure Logger ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# --- End Logger Config ---

T = TypeVar('T', bound='BaseModel')

# --- Utility function to extract location (Keep as is) ---
def extract_location_from_text(text: str, default: str = "Unknown Location") -> str:
    # ... (function code remains the same) ...
    patterns = [ r'(?:lived|resided|based)\s+in\s+([\w\s]+(?:,\s*[\w\s]+)*)', r'moved\s+to\s+([\w\s]+(?:,\s*[\w\s]+)*)' ]
    for pattern in patterns: # ... (rest of extraction logic) ...
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            location = match.group(1).strip().split(', where')[0].split(', which')[0].strip()
            if location.endswith(('.', ',', ';')): location = location[:-1].strip()
            if location: logger.debug(f"Extracted location '{location}' from text."); return location
    logger.warning(f"Could not extract specific location from text snippet: '{text[:100]}...'")
    return default

# --- Helper Function (Keep as is) ---
async def _call_llm_and_get_validated_data( llm_service: LLMService, prompt: str, response_model: Type[T], operation_description: str ) -> Optional[Dict]:
     # ... (function code remains the same) ...
    try:
        response_dict = await llm_service.generate_content( prompt=prompt, response_model=response_model )
        if not response_dict: logger.error(f"LLM call for '{operation_description}' returned None/empty."); return None
        if "error" in response_dict: error_msg = response_dict["error"]; # ... (error logging) ...; logger.error(f"LLM error during '{operation_description}': {error_msg}"); return None
        logger.debug(f"Success for '{operation_description}'."); return response_dict
    except Exception as e: logger.error(f"Exception calling LLM for '{operation_description}': {e}", exc_info=True); return None


async def generate_initial_relationships( llm_service: LLMService, persona_details_str: str ) -> Optional[Dict[str, List[Dict]]]:
    """Generates the initial family structure (parents, siblings)."""
    # ... (code remains the same) ...
    logger.info("Generating initial relationship structure...")
    prompt = f"""
Based on the following persona details: {persona_details_str}
Establish a plausible immediate family structure: parents and any siblings.
Respond ONLY with JSON: {{"parents": [{{"name": "...", "relationship": "...", "details": "..."}}], "siblings": [...]}}
"""
    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, InitialRelationshipsResponse, "initial relationships" )
    if validated_data and "parents" in validated_data and "siblings" in validated_data:
         logger.info(f"Generated initial relationships: {len(validated_data['parents'])} parents, {len(validated_data['siblings'])} siblings.")
         return validated_data
    else:
        logger.error("Failed validated initial relationships or missing keys.");
        return None

# --- Update generate_yearly_summaries (Prompt & Return Type) ---
async def generate_yearly_summaries(
    llm_service: LLMService,
    persona_details_str: str,
    initial_relationships_str: str,
    birth_year: int,
    last_year_to_generate: int,
    current_year: int
) -> Optional[Tuple[Dict[int, Tuple[str, Optional[str]]], int, int]]: # <<< RETURN TYPE CHANGED
    """
    Generates yearly summaries, asking for LOCATION in JSON output. Establishes birthday.
    Returns tuple: (summaries_dict[year -> (summary, location)], birth_month, birth_day)
    """
    logger.info(f"Generating ENRICHED yearly summaries with LOCATION JSON ({birth_year}-{last_year_to_generate}) & establishing birthday...")

    current_year_prompt_addition = ""
    if last_year_to_generate == current_year:
        current_year_prompt_addition = f"\n*   For the final year ({current_year}), summarize events *up to the present day*."

    # --- Enhanced Prompt asking for Location in JSON ---
    prompt = f"""
Persona details: {persona_details_str}
Initial family: {initial_relationships_str}
Born in {birth_year}. Summaries needed up to end of {last_year_to_generate}.

**Instructions:**

1.  **Establish Birthday:** Choose plausible birth month/day for {birth_year}.
2.  **Generate Yearly Summaries:** Provide detailed summaries (min 8-10 sentences each) for **each CALENDAR year** from {birth_year} to {last_year_to_generate}.

    **Guidance for Detail and Enrichment (Apply to each year's summary):**
    *   **Life Stage & Friendships:** Consider age, development, social life (friends critical!).
    *   **Personal Arc:** Weave in persona details/traits.
    *   **Location:** Determine the primary location for the year (infer/track moves).
    *   **Contextual Flavor:** Include 1-2 major world/regional events/trends for the year.
    *   **Integrate Context:** Plausibly connect world events to persona.
    *   **"Quieter" Years:** Include routines, relationships, reflections.
    *   **Elaborate:** Add descriptive detail.
    {current_year_prompt_addition}

**Output Format:**

Respond ONLY with a valid JSON object containing three top-level keys: "birth_month" (int), "birth_day" (int), and "summaries".
The "summaries" value must be a JSON array of objects. **Each object MUST contain THREE keys:**
- "year": integer (the calendar year).
- "location": string (The primary city/region the persona lived in this year. Be specific, e.g., "Cambridge, UK", "Manchester", "Rural Ohio"). <<< LOCATION FIELD MANDATORY
- "summary": string (The detailed narrative summary for the year, 8-10+ sentences).

Example structure snippet:
{{
  "birth_month": 7,
  "birth_day": 22,
  "summaries": [
    {{ "year": {birth_year}, "location": "Cambridge, UK", "summary": "Born July 22nd... [rest of summary]" }},
    {{ "year": {birth_year + 18}, "location": "London", "summary": "Moved to attend university... [rest of summary]" }},
    {{ "year": {last_year_to_generate}, "location": "Manchester", "summary": "Continued working... [rest of summary]" }}
  ]
}}
Ensure every year from {birth_year} to {last_year_to_generate} is included.
"""

    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, YearlySummariesResponse, f"ENRICHED yearly summaries with LOCATION JSON ({birth_year}-{last_year_to_generate})" )

    # --- Processing Logic (Handle new tuple structure) ---
    if validated_data and "summaries" in validated_data and "birth_month" in validated_data and "birth_day" in validated_data:
        birth_month = validated_data["birth_month"]; birth_day = validated_data["birth_day"]
        logger.info(f"Established Birthday: {birth_year}-{birth_month:02d}-{birth_day:02d}")

        summaries_list = validated_data["summaries"]
        # <<< Store (summary, location) tuple >>>
        summaries_dict: Dict[int, Tuple[str, Optional[str]]] = {}
        processed_years = set(); valid = True

        for item_dict in summaries_list: # Expect list of dicts now from Pydantic model
             # Pydantic model YearSummary now handles validation of year, summary, location keys/types
             if not isinstance(item_dict, dict): # Should be caught by Pydantic, but extra check
                 logger.warning(f"Unexpected item type yearly: {type(item_dict)}"); valid = False; break
             year_num = item_dict.get("year")
             summary_text = item_dict.get("summary")
             location_text = item_dict.get("location") # <<< Get location

             # Basic range check (types checked by Pydantic)
             if isinstance(year_num, int) and birth_year <= year_num <= last_year_to_generate:
                 if year_num in processed_years: logger.warning(f"Duplicate year {year_num}."); valid = False; break 
                 summaries_dict[year_num] = (summary_text, location_text) # Store tuple
                 processed_years.add(year_num)
             else: logger.warning(f"Invalid content yearly: {item_dict}, Year {year_num} range?"); valid = False; break
        if not valid: return None
        expected_count = last_year_to_generate - birth_year + 1
        if not summaries_dict: logger.warning(f"No valid yearly summaries."); return None
        logger.info(f"Processed {len(summaries_dict)} ENRICHED yearly summaries with location.")
        if len(summaries_dict) < expected_count * 0.8: logger.warning(f"Count low for yearly.")
        # <<< Return tuple including the new dict structure >>>
        return summaries_dict, birth_month, birth_day
    else:
        logger.error("Failed validated ENRICHED yearly data or missing keys.");
        return None

# --- Update generate_monthly_summaries_for_year (Prompt & Return Type) ---
async def generate_monthly_summaries_for_year(
    llm_service: 'LLMService',
    persona_details_str: str,
    year: int,
    yearly_summary: str, # The summary text part
    location_for_year: Optional[str], # <<< Explicit location passed in
    current_year: int,
    current_month: int,
    ddgs_instance: DDGS
) -> Optional[Dict[int, Tuple[str, Optional[str]]]]: # <<< RETURN TYPE CHANGED
    """
    Generates monthly summaries, using provided location, performing search,
    and asking for location in JSON output. Prunes future months.
    Returns dict: month -> (summary, location)
    """
    logger.info(f"Generating DDGS-SEARCH-ENRICHED monthly summaries for {year} (Location: {location_for_year})...")

    # --- Perform Web Search ---
    search_results_summary = "No search performed or failed."
    search_query = f"major world events cultural trends news {location_for_year or 'the world'} during {year}"
    logger.info(f"Performing DDGS search for year {year} context: '{search_query}'")
    try:
        search_results_list: List[Dict] = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=10)
        # ... (Format search results as before) ...
        if search_results_list: formatted_results = []; # ... format results ... ; search_results_summary = "\n".join(formatted_results)
        else: search_results_summary = "Search returned no results for the year."
        logger.debug(f"DDGS yearly context summary for {year}:\n{search_results_summary}")
    except Exception as search_err: logger.error(f"DDGS Web search failed for year {year}: {search_err}"); search_results_summary = "Search failed."
    # --- End Web Search ---

    # --- Enhanced Prompt asking for Location in JSON ---
    prompt = f"""
Persona details: {persona_details_str}
Focus calendar year: {year}. Assumed primary location this year: **{location_for_year or 'Not specified, infer reasonably'}**.
Base context from Yearly Summary: "{yearly_summary}"

**Context from Search about Year {year} (potentially relevant):**
{search_results_summary}

**Task:** Based *primarily* on the "Base context for this year" (Yearly Summary), augmented by Search Context, seasonality, and **location context ({location_for_year or 'inferred'})**, create an **in-depth narrative summary** for **each month** (1-12) of {year}.

**Integration and Elaboration Instructions:**
0.  **Location Context:** Ground descriptions in **{location_for_year or 'the inferred location'}**.
1.  **Foundation:** Elaborate on Yearly Summary themes across months.
2.  **Flesh out Details:** Include specific activities, internal state, sensory details (location-based), social interactions (friends!), routine (location-based), smaller observations.
3.  **World Context:** Subtly integrate relevant search results/yearly events.
4.  **Plausibility/Consistency:** Ensure consistency with persona, year summary, location.
5.  **Depth/Length:** Aim for **8-10 rich sentences minimum** per month.

**Output Format:**
Respond ONLY with a valid JSON object: {{"summaries": [...]}}
The "summaries" array must contain objects, **each with THREE keys:**
- "month": integer (1-12).
- "location": string (The location assumed for this month, usually "{location_for_year or 'Inferred Location'}"). <<< LOCATION FIELD MANDATORY
- "summary": string (The detailed narrative summary, 8-10+ sentences).

Example structure snippet:
{{
  "summaries": [
    {{ "month": 1, "location": "{location_for_year or 'Inferred Location'}", "summary": "January began..." }},
    {{ "month": 2, "location": "{location_for_year or 'Inferred Location'}", "summary": "February saw..." }}
  ]
}}
Include all 12 months initially; pruning happens later.
"""

    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, MonthlySummariesResponse, f"DDGS-SEARCH-ENRICHED monthly summaries year {year} ({location_for_year})" )

    # --- Processing Logic (Handle new tuple structure) ---
    if validated_data and "summaries" in validated_data:
        summaries_list = validated_data["summaries"]
        # <<< Store (summary, location) tuple >>>
        summaries_dict: Dict[int, Tuple[str, Optional[str]]] = {}
        processed_months = set(); valid = True
        for item_dict in summaries_list: # Expect list of dicts from Pydantic
            if not isinstance(item_dict, dict): logger.warning(f"Invalid item monthly: {type(item_dict)}"); valid = False; break
            month_num = item_dict.get("month")
            summary_text = item_dict.get("summary")
            location_text = item_dict.get("location") # <<< Get location

            if isinstance(month_num, int) and 1 <= month_num <= 12:
                 if month_num in processed_months: logger.warning(f"Duplicate month {month_num} ({year})."); valid = False; break
                 summaries_dict[month_num] = (summary_text, location_text) # Store tuple
                 processed_months.add(month_num)
            else: logger.warning(f"Invalid content monthly: {item_dict}"); valid = False; break
        if not valid: return None
        if not summaries_dict: logger.warning(f"No valid monthly summaries for year {year}."); return None

        # --- Pruning Logic ---
        if year == current_year:
            months_to_remove = [m for m in summaries_dict if m > current_month]
            if months_to_remove: logger.info(f"Pruning future months ({months_to_remove}) for year {year}."); # ... (delete loop) ...
            for m in months_to_remove: del summaries_dict[m]

        logger.info(f"Successfully processed {len(summaries_dict)} DDGS-SEARCH-ENRICHED monthly summaries for year {year} ({location_for_year}) (after pruning).")
        # <<< Return dict with tuples >>>
        return summaries_dict
    else:
        logger.error(f"Failed validated DDGS-SEARCH-ENRICHED monthly summaries for year {year}.");
        return None

# --- REVERT generate_daily_summaries_for_month to Bulk JSON with Location ---
async def generate_daily_summaries_for_month(
    llm_service: 'LLMService',
    persona_details_str: str,
    year: int, month: int,
    monthly_summary: str, yearly_summary: str, # yearly includes location hint
    location_for_year: Optional[str], # <<< Explicit location passed in
    current_year: int, current_month: int, current_day: int,
    ddgs_instance: DDGS
) -> Optional[Dict[int, Tuple[str, Optional[str]]]]: # <<< RETURN TYPE CHANGED
    """
    Generates daily summaries for the whole month in one call, asking for LOCATION
    in JSON. Performs ONE search for the month's context. Prunes future days.
    Returns dict: day -> (summary, location)
    """
    logger.info(f"Generating BULK JSON DDGS-SEARCH-ENRICHED daily summaries for {year}-{month:02d} (Loc: {location_for_year})...")
    month_name = f"Month {month}"; days_in_month = 31
    try: days_in_month = calendar.monthrange(year, month)[1]; month_name = calendar.month_name[month]
    except Exception as e: logger.error(f"Calendar error for {year}-{month}: {e}"); return None

    # --- Perform ONE Web Search for Monthly Context ---
    search_results_summary = "No search performed or failed."
    search_query = f"significant events news weather {location_for_year or 'general region'} during {month_name} {year}"
    logger.info(f"Performing DDGS search for {year}-{month:02d} context: '{search_query}'")
    try:
        search_results_list: List[Dict] = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=7)
        if search_results_list:
             formatted_results = []; # ... format results ...
             for i, result in enumerate(search_results_list):
                 title = result.get('title', 'No Title'); body = result.get('body', 'No Snippet'); body_snippet = body[:150] + '...' if len(body) > 150 else body
                 formatted_results.append(f"- {title}: {body_snippet}")
             search_results_summary = "\n".join(formatted_results)
        else: search_results_summary = "Search returned no results for the month."
        logger.debug(f"DDGS monthly context summary:\n{search_results_summary}")
    except Exception as search_err:
        logger.error(f"DDGS Web search failed for {year}-{month:02d}: {search_err}")
        search_results_summary = "Search failed."
    # --- End Web Search ---

    # --- Enhanced Prompt Asking for JSON including Location ---
    prompt = f"""
Persona Context: {persona_details_str}
Broader Context: Year={year} ("{yearly_summary}"), Month={month_name} ("{monthly_summary}")
Assumed Location for this month: **{location_for_year or 'Not specified, infer reasonably'}**

**Context from Search about {month_name} {year} (potentially relevant to {location_for_year or 'region'}):**
{search_results_summary}

**Task:** Generate realistic, detailed summaries (10 or more sentences each) for **each day** (1 to {days_in_month}) of {month_name}, {year}, reflecting the persona's likely experience **in their location ({location_for_year or 'inferred'})**.

**Integration Instructions:**
1.  **Consider Monthly Context:** Use the monthly summary and search context for overarching themes or events impacting the month.
2.  **Daily Variation:** Show plausible daily routines influenced by day-of-week, but add minor variations. Distribute monthly events logically across specific days.
3.  **Location:** Ground the summaries in the assumed location.
4.  **Search Impact:** If search results mention a major event *specifically assignable to certain days* within the month (and relevant to the location), reflect its impact on those days. Otherwise, use the search results for general monthly tone/context.

**Output Format:**
Respond ONLY with a valid JSON object: {{"summaries": [...]}}
The "summaries" array must contain objects, **each with THREE keys:**
- "day": integer (1-{days_in_month}).
- "location": string (The assumed location for this day, usually "{location_for_year or 'Inferred Location'}"). <<< LOCATION FIELD MANDATORY
- "summary": string (The concise 1-2 sentence summary for the day).

Example structure snippet:
{{
  "summaries": [
    {{ "day": 1, "location": "{location_for_year or 'Inferred Location'}", "summary": "Started the month..." }},
    {{ "day": 2, "location": "{location_for_year or 'Inferred Location'}", "summary": "Continued with..." }},
    // ... down to day {days_in_month}
  ]
}}
Include all {days_in_month} days initially; pruning happens later.
"""

    # Call LLM using helper (expects JSON)
    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, DailySummariesResponse, f"BULK JSON DDGS-SEARCH daily summaries {year}-{month:02d} ({location_for_year})" )

    # --- Processing Logic (Handle new tuple structure) ---
    if validated_data and "summaries" in validated_data:
        summaries_list = validated_data["summaries"]
        # <<< Store (summary, location) tuple >>>
        summaries_dict: Dict[int, Tuple[str, Optional[str]]] = {}
        processed_days = set(); valid = True
        for item_dict in summaries_list: # Expect list of dicts from Pydantic
            if not isinstance(item_dict, dict): logger.warning(f"Invalid item daily: {type(item_dict)}"); valid = False; break
            day_num = item_dict.get("day")
            summary_text = item_dict.get("summary")
            location_text = item_dict.get("location") # <<< Get location

            if isinstance(day_num, int) and 1 <= day_num <= days_in_month:
                 if day_num in processed_days: logger.warning(f"Duplicate day {day_num} ({year}-{month:02d})."); valid = False; break
                 summaries_dict[day_num] = (summary_text, location_text) # Store tuple
                 processed_days.add(day_num)
            else: logger.warning(f"Invalid content daily: {item_dict}"); valid = False; break
        if not valid: return None
        if not summaries_dict: logger.warning(f"No valid daily summaries for {year}-{month:02d}."); return None

        # --- Pruning Logic ---
        if year == current_year and month == current_month:
            days_to_remove = [d for d in summaries_dict if d > current_day]
            if days_to_remove: logger.info(f"Pruning future days ({days_to_remove}) for {year}-{month:02d}."); # ... (delete loop) ...
            for d in days_to_remove: del summaries_dict[d]

        logger.info(f"Successfully processed {len(summaries_dict)} BULK JSON DDGS-SEARCH daily summaries for {year}-{month:02d} ({location_for_year}) (after pruning).")
        # <<< Return dict with tuples >>>
        return summaries_dict
    else:
        logger.error(f"Failed validated BULK JSON DDGS-SEARCH daily summaries for {year}-{month:02d}.");
        return None

# --- Update generate_hourly_breakdown_for_day (Prompt & Return Type) ---
async def generate_hourly_breakdown_for_day(
    llm_service: LLMService,
    persona_details_str: str,
    year: int, month: int, day: int,
    daily_summary: str, # Summary text part
    monthly_summary: str, # Summary text part
    yearly_summary: str, # Summary text part
    location_for_day: Optional[str], # <<< Explicit location passed in
    current_year: int, current_month: int, current_day: int, current_hour: int
) -> Optional[Dict[int, Tuple[str, Optional[str]]]]: # <<< RETURN TYPE CHANGED
    """
    Generates hourly breakdown, asking for LOCATION in JSON output.
    Prunes future hours for the current day.
    Returns dict: hour -> (activity, location)
    """
    logger.info(f"Generating JSON hourly breakdown for {year}-{month:02d}-{day:02d} (Loc: {location_for_day})...")

    # --- Enhanced Prompt asking for Location in JSON ---
    prompt = f"""
Persona: {persona_details_str}
Focus Date: {year}-{month:02d}-{day:02d}. Assumed Location: **{location_for_day or 'Not specified, infer reasonably'}**.
Year Context: "{yearly_summary}"
Month Context: "{monthly_summary}"
Day Context (Target): "{daily_summary}"

**Task:** Generate a detailed, realistic hour-by-hour breakdown (0-23) for the Focus Date, grounded in the **Assumed Location ({location_for_day or 'inferred'})**.

**Instructions:**
1.  **Detail per Hour:** 1-2 concise sentences describing primary activity/state.
2.  **Mandatory Inclusions:** Incorporate realistic sleep cycle, meals, hygiene, work/study/leisure (from context), social interaction (if any), commute/travel (if any), breaks, chores.
3.  **Context Integration:** Hourly activities must flesh out and be consistent with Day/Month/Year context and Persona details, reflecting the **Assumed Location**.
4.  **Logical Flow & Transitions:** Ensure smooth transitions between hours.
5.  **Realism:** Plausible sequence for a real person in **{location_for_day or 'the inferred location'}**.

**Output Format:**
Respond ONLY with a valid JSON object: {{"activities": [...]}}
The "activities" array must contain objects, **each with THREE keys:**
- "hour": integer (0-23).
- "location": string (The assumed location for this hour, usually "{location_for_day or 'Inferred Location'}"). <<< LOCATION FIELD MANDATORY
- "activity": string (The 1-2 sentence description for the hour).

Example structure snippet:
{{
  "activities": [
    {{ "hour": 0, "location": "{location_for_day or 'Inferred Location'}", "activity": "Sleeping soundly." }},
    // ...
    {{ "hour": 8, "location": "{location_for_day or 'Inferred Location'}", "activity": "Preparing breakfast in the kitchen..." }},
    // ...
    {{ "hour": 23, "location": "{location_for_day or 'Inferred Location'}", "activity": "In bed, falling asleep." }}
  ]
}}
Include all 24 hours initially; pruning happens later.
"""
    # Call LLM using helper (expects JSON)
    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, HourlyBreakdownResponse, f"JSON hourly breakdown {year}-{month:02d}-{day:02d} ({location_for_day})" )

    # --- Processing Logic (Handle new tuple structure) ---
    if validated_data and "activities" in validated_data:
        activities_list = validated_data["activities"]
        # <<< Store (activity, location) tuple >>>
        activities_dict: Dict[int, Tuple[str, Optional[str]]] = {}
        processed_hours = set(); valid = True
        for item_dict in activities_list: # Expect list of dicts from Pydantic
            if not isinstance(item_dict, dict): logger.warning(f"Invalid item hourly: {type(item_dict)}"); valid = False; break
            hour_num = item_dict.get("hour")
            activity_text = item_dict.get("activity")
            location_text = item_dict.get("location") # <<< Get location

            if isinstance(hour_num, int) and 0 <= hour_num <= 23:
                if hour_num in processed_hours: logger.warning(f"Duplicate hour {hour_num} ({year}-{month:02d}-{day:02d})."); valid = False; break
                activities_dict[hour_num] = (activity_text, location_text) # Store tuple
                processed_hours.add(hour_num)
            else: logger.warning(f"Invalid content hourly: {item_dict}"); valid = False; break
        if not valid: return None
        if not activities_dict: logger.warning(f"No valid hourly activities for {year}-{month:02d}-{day:02d}."); return None

        # --- Pruning Logic ---
        if year == current_year and month == current_month and day == current_day:
            hours_to_remove = [h for h in activities_dict if h > current_hour]
            if hours_to_remove: logger.info(f"Pruning future hours ({hours_to_remove}) for {year}-{month:02d}-{day:02d}."); # ... (delete loop) ...
            for h in hours_to_remove: del activities_dict[h]

        logger.info(f"Successfully processed {len(activities_dict)} JSON hourly activities for {year}-{month:02d}-{day:02d} ({location_for_day}) (after pruning).")
        # <<< Return dict with tuples >>>
        return activities_dict
    else:
        logger.error(f"Failed validated JSON hourly activities for {year}-{month:02d}-{day:02d}.");
        return None


# --- Orchestration Function (Corrected Hourly Day Calculation) ---
async def generate_life_summary_sequentially(
    llm_service: LLMService,
    persona_details: Dict[str, Any],
    age: int
) -> Dict[str, Any]:
    """
    Orchestrates generation with locations included in models/responses.
    Handles plain text daily summaries.
    """
    # --- Setup: Date/Time and Basic Info ---
    now = datetime.now()
    today = now.date()
    current_year = now.year
    current_month = now.month
    current_day = now.day
    current_hour = now.hour
    birth_year = current_year - age

    logger.info(f"Starting LOC-AWARE generation (Age: {age}, Current: {now.strftime('%Y-%m-%d %H:%M')}, Birth: {birth_year}).")

    # --- Initialize main data structure ---
    life_summary = {
        "persona_details": persona_details, "age": age, "birth_year": birth_year,
        "birth_month": None, "birth_day": None, "initial_relationships": None,
        "yearly_summaries": {},    # Stores Dict[int, Tuple[str, Optional[str]]] -> (summary, location)
        "monthly_summaries": {},   # Stores Dict[int, Dict[int, Tuple[str, Optional[str]]]] -> year -> month -> (summary, location)
        "daily_summaries": {},     # Stores Dict[int, Dict[int, Dict[int, str]]] -> year -> month -> day -> summary_text
        "hourly_breakdowns": {}    # Stores Dict[int, Dict[int, Dict[int, Dict[int, Tuple[str, Optional[str]]]]]] -> year -> month -> day -> hour -> (activity, location)
    }

    # --- Format persona details for prompts ---
    details_list = [
        f"Name: {persona_details.get('Name', 'N/A')}",
        f"Age: {age}",
        f"Born Year: ~{birth_year}",
        f"Occupation: {persona_details.get('Occupation', 'N/A')}",
        f"Location: {persona_details.get('Current location', 'N/A')}",
        f"Personality: {str(persona_details.get('Personality_Traits', 'N/A'))[:100]}..."
    ]
    persona_details_str = ", ".join(details_list)

    # --- Initialize Search Tool ---
    ddgs_search_tool = DDGS()
    # --- End Setup ---

    # --- 0. Generate Initial Relationships ---
    print("\n" + "*"*15 + " GENERATING INITIAL RELATIONSHIPS " + "*"*15)
    initial_relationships_data = await generate_initial_relationships(llm_service, persona_details_str)
    if initial_relationships_data:
        life_summary["initial_relationships"] = initial_relationships_data
        rel_parts = []
        if initial_relationships_data.get('parents'):
            rel_parts.append("Parents: " + ", ".join([f"{p.get('name', '?')} ({p.get('details', '?')})" for p in initial_relationships_data['parents']]))
        if initial_relationships_data.get('siblings'):
            rel_parts.append("Siblings: " + ", ".join([f"{s.get('name', '?')} ({s.get('relationship', '?')}, {s.get('details', '?')})" for s in initial_relationships_data['siblings']]))
        initial_relationships_str = "; ".join(rel_parts) if rel_parts else "No family data generated."
    else:
        logger.warning("Failed initial relationships.")
        initial_relationships_str = "No family data generated."
        life_summary["initial_relationships"] = {}
    logger.info(f"Relationship context: {initial_relationships_str}")
    # --- End Step 0 ---

    # --- 1. Generate Yearly Summaries & GET BIRTHDAY ---
    print("\n" + "*"*15 + " GENERATING YEARLY SUMMARIES (Location Aware) " + "*"*15)
    yearly_summaries: Dict[int, Tuple[str, Optional[str]]] = {}
    birth_month, birth_day = None, None
    total_summarized_years = 0
    last_year_to_generate = current_year
    if birth_year <= last_year_to_generate:
        yearly_result = await generate_yearly_summaries(llm_service, persona_details_str, initial_relationships_str, birth_year, last_year_to_generate, current_year)
        if yearly_result:
            yearly_summaries, birth_month, birth_day = yearly_result
            life_summary["birth_month"] = birth_month
            life_summary["birth_day"] = birth_day
            life_summary["yearly_summaries"] = yearly_summaries
            total_summarized_years = len(yearly_summaries)
            logger.info(f"Stored birthday & {total_summarized_years} yearly summaries with locations.")
        else:
            logger.error("Failed yearly summaries/birthday. Cannot proceed reliably without yearly context.")
            return life_summary # Return potentially incomplete summary
    else:
        logger.warning(f"Birth year {birth_year} > current year {last_year_to_generate}. Skipping generation.")
        return life_summary # Nothing to generate
    # --- End Step 1 ---

    # --- 2. Generate Monthly Summaries ---
    print("\n" + "*"*15 + " GENERATING MONTHLY SUMMARIES (Location Aware) " + "*"*15)
    years_for_monthly = []
    if total_summarized_years > 0:
        available_years = sorted(yearly_summaries.keys())
        monthly_count = math.ceil(total_summarized_years / 3)
        monthly_count = min(monthly_count, total_summarized_years) # Don't request more than available
        start_index = len(available_years) - monthly_count
        years_for_monthly = available_years[start_index:] if len(available_years) >= monthly_count else available_years
        logger.info(f"Generating monthly summaries for {len(years_for_monthly)} years: {years_for_monthly}")
    else:
        logger.warning("No summarized years available, skipping monthly generation.")

    for year_to_process in years_for_monthly:
        if year_to_process not in yearly_summaries:
            logger.warning(f"Year {year_to_process} requested for monthly not found in yearly summaries. Skipping.")
            continue
        summary_text, location_text = yearly_summaries[year_to_process]
        monthly_data = await generate_monthly_summaries_for_year(
            llm_service, persona_details_str, year_to_process,
            summary_text, location_text, # Pass location
            current_year, current_month, ddgs_search_tool
        )
        if monthly_data:
            life_summary["monthly_summaries"][year_to_process] = monthly_data
    # --- End Step 2 ---

    # --- 3. Generate Daily Summaries (Day-by-Day Plain Text) ---
    print("\n" + "*"*15 + " GENERATING DAILY SUMMARIES (Location Aware, Day-by-Day) " + "*"*15)
    months_for_daily: List[Tuple[int, int]] = []
    year_month_current = (current_year, current_month)
    first_of_this_month = today.replace(day=1)
    last_month_end_date = first_of_this_month - timedelta(days=1)
    year_month_previous = (last_month_end_date.year, last_month_end_date.month)

    # Add previous month if not before birth year
    if year_month_previous[0] >= birth_year:
        # Check against birth date if birth year is the same
        if birth_month and birth_day and year_month_previous[0] == birth_year:
             if year_month_previous[1] >= birth_month:
                  months_for_daily.append(year_month_previous)
             else:
                  logger.info(f"Previous month ({year_month_previous}) is before birth month in birth year. Skipping.")
        else:
             months_for_daily.append(year_month_previous)
    else:
        logger.info(f"Previous month ({year_month_previous}) is before birth year. Skipping.")

    # Add current month if not before birth year/month and not same as previous
    if year_month_current[0] >= birth_year:
         if birth_month and birth_day and year_month_current[0] == birth_year:
              if year_month_current[1] >= birth_month:
                   if year_month_current != year_month_previous:
                       months_for_daily.append(year_month_current)
              else:
                   logger.info(f"Current month ({year_month_current}) is before birth month in birth year. Skipping.")
         else:
              if year_month_current != year_month_previous:
                  months_for_daily.append(year_month_current)
    else:
         logger.info(f"Current month ({year_month_current}) appears to be before birth year. Skipping.")

    months_for_daily.sort()
    logger.info(f"Generating daily summaries for months: {months_for_daily}")

    for year_d, month_d in months_for_daily:
        yearly_data_tuple = yearly_summaries.get(year_d)
        if not yearly_data_tuple:
            logger.warning(f"Skipping daily {year_d}-{month_d:02d}: Missing yearly tuple.")
            continue
        yearly_context_daily, location_for_year = yearly_data_tuple

        monthly_data_tuple = life_summary["monthly_summaries"].get(year_d, {}).get(month_d)
        monthly_context_daily = monthly_data_tuple[0] if monthly_data_tuple else "N/A"

        daily_data = await generate_daily_summaries_for_month(
            llm_service, persona_details_str, year_d, month_d,
            monthly_context_daily, yearly_context_daily, location_for_year, # Pass location
            current_year, current_month, current_day,
            ddgs_search_tool
        )
        if daily_data:
            life_summary["daily_summaries"].setdefault(year_d, {})[month_d] = daily_data # Store plain Dict[int, str]
    # --- End Step 3 ---

    # --- 4. Generate Hourly Breakdowns (Handles Plain Text Daily Context) ---
    print("\n" + "*"*15 + " GENERATING HOURLY BREAKDOWNS (Location Aware) " + "*"*15)
    days_for_hourly: List[Tuple[int, int, int]] = []
    for i in range(7):
        target_date = today - timedelta(days=i)
        if target_date.year >= birth_year:
            if birth_month and birth_day:
                # Ensure birth date calculation uses the target_date's year
                try:
                    birth_date_this_year = date(target_date.year, birth_month, birth_day)
                    if target_date.year == birth_year and target_date < birth_date_this_year:
                        continue
                except ValueError: # Handle invalid birth date combination (e.g., Feb 29 in non-leap year)
                    logger.warning(f"Could not form valid birth date {birth_month}-{birth_day} for year {target_date.year}, proceeding without exact day check.")
                    # If birth year, still check month at least
                    if target_date.year == birth_year and target_date.month < birth_month:
                        continue
                days_for_hourly.append((target_date.year, target_date.month, target_date.day))
            else:
                break # Stop if we go before birth year

        days_for_hourly.reverse() 
    logger.info(f"Generating hourly breakdown for last {len(days_for_hourly)} days (up to 7): {days_for_hourly}")

    for year, month, day in days_for_hourly:
         yearly_data_tuple = yearly_summaries.get(year)
         yearly_context_hr = yearly_data_tuple[0] if yearly_data_tuple else "N/A"
         location_for_year_hr = yearly_data_tuple[1] if yearly_data_tuple else None

         monthly_data_tuple = life_summary["monthly_summaries"].get(year, {}).get(month)
         monthly_context_hr = monthly_data_tuple[0] if monthly_data_tuple else "N/A"
         location_for_month_hr = monthly_data_tuple[1] if monthly_data_tuple and monthly_data_tuple[1] else location_for_year_hr

         # Fetch plain text daily summary
         daily_summary_text = life_summary["daily_summaries"].get(year, {}).get(month, {}).get(day)
         location_for_day_hr = location_for_month_hr # Best guess for day's location

         if daily_summary_text is None:
             logger.warning(f"Daily summary for {year}-{month:02d}-{day:02d} missing. Attempting on-demand...")
             if yearly_context_hr == "N/A":
                 logger.error(f"Cannot gen on-demand daily: Missing yearly {year}. Skip hourly.")
                 continue
             # Generate monthly on demand if needed
             if monthly_context_hr == "N/A":
                 logger.info(f"Monthly summary for {year}-{month:02d} missing. Generating on-demand...")
                 monthly_data_ondemand = await generate_monthly_summaries_for_year(
                     llm_service, persona_details_str, year, yearly_context_hr,
                     location_for_year_hr, current_year, current_month, ddgs_search_tool
                 )
                 if monthly_data_ondemand:
                     life_summary["monthly_summaries"].setdefault(year, {})[month] = monthly_data_ondemand
                     monthly_data_tuple = monthly_data_ondemand.get(month)
                     monthly_context_hr = monthly_data_tuple[0] if monthly_data_tuple else "N/A"
                     location_for_month_hr = monthly_data_tuple[1] if monthly_data_tuple and monthly_data_tuple[1] else location_for_year_hr
                     logger.info(f"Success on-demand monthly {year}-{month:02d}.")
                 else:
                     logger.error(f"Failed on-demand monthly {year}-{month:02d}.")
                     monthly_context_hr = "N/A" # Ensure it's marked N/A

             # Generate daily on demand (plain text)
             logger.info(f"Generating on-demand daily summary (plain text) for {year}-{month:02d}-{day:02d}...")
             daily_data_ondemand_dict = await generate_daily_summaries_for_month(
                 llm_service, persona_details_str, year, month,
                 monthly_context_hr, yearly_context_hr, location_for_year_hr,
                 current_year, current_month, current_day, ddgs_search_tool
             )
             if daily_data_ondemand_dict and day in daily_data_ondemand_dict:
                 daily_summary_text = daily_data_ondemand_dict[day]
                 life_summary["daily_summaries"].setdefault(year, {}).setdefault(month, {})[day] = daily_summary_text
                 location_for_day_hr = location_for_month_hr # Still use inferred location
                 logger.info("Success on-demand daily.")
             else:
                 logger.error(f"Failed on-demand daily {year}-{month:02d}-{day:02d}. Skipping hourly.")
                 continue

         # Generate hourly breakdown
         hourly_data = await generate_hourly_breakdown_for_day(
             llm_service, persona_details_str, year, month, day,
             daily_summary_text, monthly_context_hr, yearly_context_hr,
             location_for_day_hr, # Pass inferred location
             current_year, current_month, current_day, current_hour
         )
         if hourly_data:
             life_summary["hourly_breakdowns"].setdefault(year, {}).setdefault(month, {})[day] = hourly_data
    # --- End Step 4 ---

    logger.info("Sequential life summary generation finished.")
    return life_summary

async def generate_random_persona(llm_service: LLMService) -> Optional[Dict[str, Any]]:
    """Generates a random plausible persona including age using the LLM."""
    logger.info("Generating random persona...")
    prompt = f"""
Create a detailed and plausible fictional persona profile. Ensure the details are consistent and somewhat interesting.

**Required Fields:**
*   Name: Full name.
*   Age: A plausible current integer age between 18 and 45. 
*   Occupation: Current occupation (or student status, consistent with age).
*   Current_location: City, State/Country format.
*   Personality_Traits: A list of 3-6 descriptive adjectives.
*   Birthplace: City, State/Country format.
*   Education: Highest level achieved or current status (consistent with age).

Respond ONLY with a valid JSON object matching this structure:
{{
  "Name": "...",
  "Age": ..., # <<< Added Age
  "Occupation": "...",
  "Current_location": "...",
  "Personality_Traits": ["...", "...", "..."],
  "Birthplace": "...",
  "Education": "..."
}}
Ensure Age is an integer.
"""
    validated_data = await _call_llm_and_get_validated_data(
        llm_service,
        prompt,
        PersonaDetailsResponse, # Validate against the updated model
        "random persona generation"
    )

    if validated_data:
        logger.info(f"Successfully generated random persona: {validated_data.get('Name', 'Unknown')}, Age: {validated_data.get('Age', 'N/A')}")
        return validated_data
    else:
        logger.error("Failed to generate or validate random persona.")
        return None

# --- Main Execution Block (Restored LLM Init, Generated Age) ---
if __name__ == "__main__":
    async def run_test():
        logger.info("Starting life summary generation test...")
        api_key = os.getenv("GOOGLE_API_KEY")
        llm_service = None

        try:
            llm_service = LLMService(api_key=api_key)
            logger.info("Real LLMService initialized successfully.")
        except Exception as e:
            logger.critical(f"Error initializing real LLMService: {e}. Cannot proceed.", exc_info=True)
            return
        # --- Generate Random Persona (including Age) ---
        print("\n" + "*"*15 + " GENERATING RANDOM PERSONA " + "*"*15)
        generated_persona = await generate_random_persona(llm_service)

        if not generated_persona:
            logger.error("Could not generate persona. Using fallback.")
            # Fallback persona *including age*
            generated_persona = {
                "Name": "Alex Default",
                "Age": 35, # Add default age
                "Occupation": "Librarian",
                "Current_location": "Portland, Oregon",
                "Personality_Traits": ["Quiet", "Organized", "Helpful"],
                "Birthplace": "Chicago, Illinois",
                "Education": "Master of Library Science",
            }
            sample_age = generated_persona["Age"] # Use fallback age
            print(f"Using Fallback Persona:\n{json.dumps(generated_persona, indent=2)}")
        else:
             print(f"Generated Persona:\n{json.dumps(generated_persona, indent=2)}")
             # <<< Extract age from generated persona >>>
             sample_age = generated_persona.get("Age")
             if not isinstance(sample_age, int) or sample_age < 1:
                 logger.error(f"Invalid age '{sample_age}' generated. Falling back to age 30.")
                 sample_age = 30 # Fallback if age is missing or invalid type

        logger.info(f"Using persona '{generated_persona.get('Name')}' with generated age: {sample_age}")
        # -----------------------------------------

        # --- Run Life Summary Generation ---
        print("\n" + "*"*15 + f" GENERATING LIFE SUMMARY (Age: {sample_age}) " + "*"*15)
        life_data = await generate_life_summary_sequentially(
            llm_service=llm_service,
            persona_details=generated_persona, # Use the generated (or fallback) persona
            age=sample_age # Use the extracted or fallback age
        )
        # -------------------------------

        # --- Save Results ---
        output_filename = f"life_summary_{generated_persona.get('Name', 'Unknown').replace(' ', '_')}_{sample_age}_generated.json"
        logger.info(f"Generation complete. Saving results to {output_filename}")
        try:
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(life_data, f, indent=2, ensure_ascii=False)
            logger.info("Results saved successfully.")
        except TypeError as e:
            logger.error(f"TypeError saving results: {e}. Attempting simplified save.")
            try:
                simplified_data = json.loads(json.dumps(life_data, default=str))
                with open(output_filename.replace(".json", "_simplified.json"), 'w', encoding='utf-8') as f:
                    json.dump(simplified_data, f, indent=2, ensure_ascii=False)
                logger.info("Saved simplified version.")
            except Exception as dump_err:
                 logger.error(f"Could not save simplified data: {dump_err}")
        except Exception as e:
            logger.error(f"Error saving results: {e}", exc_info=True)

    # Removed import random as it's no longer needed for age
    asyncio.run(run_test())