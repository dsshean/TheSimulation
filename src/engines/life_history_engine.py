import re
import math
import logging
import asyncio
import json
import calendar
import os
from datetime import date, timedelta, datetime
from typing import Dict, Any, Optional, List, Tuple, Type, TypeVar, Union

# Assuming these are importable from the src directory level
from ..llm_service import LLMService
from ..models import (
    InitialRelationshipsResponse, Person, YearlySummariesResponse,
    MonthlySummariesResponse, DailySummariesResponse, HourlyBreakdownResponse,
    YearSummary, MonthSummary, DaySummary, HourActivity, PersonaDetailsResponse
)
# Assuming duckduckgo_search is installed
from duckduckgo_search import DDGS

# Configure logger for this module
logger = logging.getLogger(__name__)

T = TypeVar('T', bound='BaseModel')

# --- Utility function moved here ---
def extract_location_from_text(text: str, default: str = "Unknown Location") -> str:
    patterns = [ r'(?:lived|resided|based)\s+in\s+([\w\s]+(?:,\s*[\w\s]+)*)', r'moved\s+to\s+([\w\s]+(?:,\s*[\w\s]+)*)' ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            location = match.group(1).strip().split(', where')[0].split(', which')[0].strip()
            if location.endswith(('.', ',', ';')): location = location[:-1].strip()
            if location: logger.debug(f"Extracted location '{location}' from text."); return location
    logger.warning(f"Could not extract specific location from text snippet: '{text[:100]}...'")
    return default

# --- Helper Function moved here ---
async def _call_llm_and_get_validated_data( llm_service: LLMService, prompt: str, response_model: Type[T], operation_description: str ) -> Optional[Dict]:
     try:
         # Consider passing max_tokens if needed, e.g., from a config or engine setting
         response_dict = await llm_service.generate_content( prompt=prompt, response_model=response_model)
         if not response_dict: logger.error(f"LLM call for '{operation_description}' returned None/empty."); return None
         if "error" in response_dict: error_msg = response_dict["error"]; logger.error(f"LLM error during '{operation_description}': {error_msg}"); return None
         logger.debug(f"Success for '{operation_description}'."); return response_dict
     except Exception as e: logger.error(f"Exception calling LLM for '{operation_description}': {e}", exc_info=True); return None

# --- Persona Generation Function moved here ---
async def generate_random_persona(llm_service: LLMService) -> Optional[Dict[str, Any]]:
    """Generates a random plausible persona including age using the LLM."""
    logger.info("Generating random persona...")
    prompt = f"""
Create a detailed and plausible fictional persona profile. Ensure the details are consistent and somewhat interesting.
**Required Fields:** Name, Age (18-85 integer), Occupation (consistent w/ age), Current_location (City, State/Country), Personality_Traits (list of 3-6 adj.), Birthplace (City, State/Country), Education (consistent w/ age).
Respond ONLY with valid JSON: {{"Name": "...", "Age": ..., "Occupation": "...", "Current_location": "...", "Personality_Traits": [...], "Birthplace": "...", "Education": "..."}}
"""
    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, PersonaDetailsResponse, "random persona generation" )
    if validated_data: logger.info(f"Generated persona: {validated_data.get('Name', 'Unknown')}, Age: {validated_data.get('Age', 'N/A')}"); return validated_data
    else: logger.error("Failed to generate/validate persona."); return None

# --- Relationship Generation moved here ---
async def generate_initial_relationships( llm_service: LLMService, persona_details_str: str ) -> Optional[Dict[str, List[Dict]]]:
    """Generates the initial family structure."""
    logger.info("Generating initial relationship structure...")
    prompt = f"""Based on: {persona_details_str}. Establish plausible parents and siblings. Respond ONLY with JSON: {{"parents": [...], "siblings": [...]}}"""
    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, InitialRelationshipsResponse, "initial relationships" )
    if validated_data and "parents" in validated_data and "siblings" in validated_data: logger.info(f"Generated relationships."); return validated_data
    else: logger.error("Failed validation/keys for initial relationships."); return None

# --- Yearly Generation moved here ---
async def generate_yearly_summaries(
    llm_service: LLMService, persona_details_str: str, initial_relationships_str: str,
    birth_year: int, last_year_to_generate: int, current_year: int
) -> Optional[Tuple[Dict[int, Tuple[str, Optional[str]]], int, int]]:
    """Generates yearly summaries with location."""
    logger.info(f"Generating yearly summaries w/ location JSON ({birth_year}-{last_year_to_generate})...")
    current_year_prompt_addition = f"\n* For {current_year}, summarize events up to the present day." if last_year_to_generate == current_year else ""
    prompt = f"""
Persona: {persona_details_str} | Family: {initial_relationships_str} | Born: {birth_year} | Summarize to: {last_year_to_generate}.
**Instructions:** 1. Establish plausible birth month/day. 2. Generate detailed summaries (8-10+ sentences/year) for each year {birth_year}-{last_year_to_generate}.
**Guidance:** Include Life Stage/Friends, Personal Arc, Location (MANDATORY in JSON), Contextual Flavor (world events), Integration, Quieter Years, Elaboration.{current_year_prompt_addition}
**Output Format:** ONLY valid JSON: {{"birth_month": int, "birth_day": int, "summaries": [{{"year": int, "location": str, "summary": str}}]}}
"""
    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, YearlySummariesResponse, f"Yearly summaries w/ location JSON ({birth_year}-{last_year_to_generate})" )
    if validated_data and "summaries" in validated_data and "birth_month" in validated_data and "birth_day" in validated_data:
        birth_month = validated_data["birth_month"]; birth_day = validated_data["birth_day"]; logger.info(f"Established Birthday: {birth_year}-{birth_month:02d}-{birth_day:02d}")
        summaries_list = validated_data["summaries"]; summaries_dict: Dict[int, Tuple[str, Optional[str]]] = {}; processed_years = set(); valid = True
        for item_dict in summaries_list:
             year_num = item_dict.get("year"); summary_text = item_dict.get("summary"); location_text = item_dict.get("location")
             if isinstance(year_num, int) and birth_year <= year_num <= last_year_to_generate:
                 if year_num in processed_years: logger.warning(f"Duplicate year {year_num}."); valid = False; break
                 summaries_dict[year_num] = (summary_text, location_text); processed_years.add(year_num)
             else: logger.warning(f"Invalid yearly content: {item_dict}"); valid = False; break
        if not valid or not summaries_dict: logger.warning("No valid yearly summaries."); return None
        logger.info(f"Processed {len(summaries_dict)} yearly summaries."); return summaries_dict, birth_month, birth_day
    else: logger.error("Failed validated yearly data/keys."); return None

# --- Monthly Generation moved here ---
async def generate_monthly_summaries_for_year(
    llm_service: 'LLMService', persona_details_str: str, year: int, yearly_summary: str,
    location_for_year: Optional[str], current_year: int, current_month: int, ddgs_instance: DDGS
) -> Optional[Dict[int, Tuple[str, Optional[str]]]]:
    """Generates monthly summaries with location."""
    logger.info(f"Generating monthly summaries for {year} (Loc: {location_for_year})...")
    search_results_summary = "No search results."; search_query = f"major world events news {location_for_year or 'world'} {year}"
    try:
        results = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=10)
        if results: formatted = [f"- {r.get('title', 'NT')}: {r.get('body', 'NS')[:100]}..." for r in results]; search_results_summary = "\n".join(formatted)
    except Exception as e: logger.error(f"DDGS search failed for year {year}: {e}")
    prompt = f"""
Persona: {persona_details_str} | Year: {year} | Location: **{location_for_year or 'Not specified'}** | Yearly Summary: "{yearly_summary}"
**Search Context ({year}):**\n{search_results_summary}
**Task:** Create detailed summaries (8-10+ sentences/month) for **each month** (1-12) of {year}, grounded in location.
**Instructions:** Elaborate on yearly themes, include activities, internal state, sensory details (location!), social (friends!), routine, smaller moments. Integrate search context subtly.
**Output Format:** ONLY valid JSON: {{"summaries": [{{"month": int, "location": str, "summary": str}}]}} (location usually "{location_for_year or 'Inferred Location'}"). All 12 months.
"""
    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, MonthlySummariesResponse, f"Monthly summaries year {year} ({location_for_year})" )
    if validated_data and "summaries" in validated_data:
        summaries_list = validated_data["summaries"]; summaries_dict: Dict[int, Tuple[str, Optional[str]]] = {}; processed_months = set(); valid = True
        for item_dict in summaries_list:
            month_num = item_dict.get("month"); summary_text = item_dict.get("summary"); location_text = item_dict.get("location")
            if isinstance(month_num, int) and 1 <= month_num <= 12:
                 if month_num in processed_months: logger.warning(f"Duplicate month {month_num}."); valid = False; break
                 summaries_dict[month_num] = (summary_text, location_text); processed_months.add(month_num)
            else: logger.warning(f"Invalid monthly content: {item_dict}"); valid = False; break
        if not valid or not summaries_dict: logger.warning(f"No valid monthly summaries for {year}."); return None
        if year == current_year: # Pruning
            months_to_remove = [m for m in summaries_dict if m > current_month]
            if months_to_remove: logger.info(f"Pruning months {months_to_remove} for {year}.");
            for m in months_to_remove: del summaries_dict[m]
        logger.info(f"Processed {len(summaries_dict)} monthly summaries for {year}."); return summaries_dict
    else: logger.error(f"Failed validated monthly summaries for {year}."); return None

# --- Daily Generation (Day-by-Day Plain Text) moved here ---
async def generate_daily_summaries_for_month(
    llm_service: 'LLMService', persona_details_str: str, year: int, month: int, monthly_summary: str,
    yearly_summary: str, location_for_year: Optional[str], current_year: int, current_month: int,
    current_day: int, ddgs_instance: DDGS
) -> Optional[Dict[int, str]]:
    """Generates daily summaries DAY BY DAY (plain text)."""
    logger.info(f"Generating DAY-BY-DAY PlainText daily summaries for {year}-{month:02d} (Loc: {location_for_year})...")
    month_name = f"Month {month}"; days_in_month = 31;
    try: days_in_month = calendar.monthrange(year, month)[1]; month_name = calendar.month_name[month]
    except Exception: days_in_month = 30 # Fallback
    summaries_dict: Dict[int, str] = {}; loc = location_for_year or "region"; logger.info(f"Using location '{loc}' for daily gen {year}-{month:02d}.")
    for day_num in range(1, days_in_month + 1):
        if year == current_year and month == current_month and day_num > current_day: break
        current_date_str = f"{year}-{month:02d}-{day_num:02d}"; logger.debug(f"Processing day: {current_date_str}")
        search_results_summary = "No search results."; search_query = f"major events news weather {loc} on {current_date_str}"
        try:
            results = await asyncio.to_thread(ddgs_instance.text, keywords=search_query, max_results=5)
            if results: formatted = [f"{i+1}. {r.get('title', 'NT')}: {r.get('body', 'NS')[:100]}..." for i, r in enumerate(results)]; search_results_summary = "\n".join(formatted)
        except Exception as e: logger.error(f"DDGS search failed for {current_date_str}: {e}")
        prompt_for_day = f"""
Persona: {persona_details_str} | Context: Y={year}, M={month_name} ("{monthly_summary}") | Location: **{loc}** | Target Date: {current_date_str}
**Search Context:**\n{search_results_summary}
**Task:** Generate ONLY a concise summary (1-2 sentences) for {current_date_str} in {loc}. Integrate relevant search results if impactful for date/location, otherwise base on persona/context/day-of-week. Output ONLY the summary string.
"""
        try:
            response_data = await llm_service.generate_content(prompt=prompt_for_day) # Expect plain text
            if response_data and "text" in response_data and response_data["text"].strip():
                 day_summary_text = response_data["text"].strip()[:500] # Trim long ones
                 summaries_dict[day_num] = day_summary_text; logger.debug(f"Stored summary for {current_date_str}")
            else: logger.error(f"LLM daily call failed/empty for {current_date_str}"); continue
        except Exception as e: logger.error(f"LLM exception for day {current_date_str}: {e}", exc_info=True); continue
    if not summaries_dict: logger.warning(f"No valid daily summaries for {year}-{month:02d}."); return None
    logger.info(f"Processed {len(summaries_dict)} daily summaries for {year}-{month:02d}."); return summaries_dict

# --- Hourly Generation moved here ---
async def generate_hourly_breakdown_for_day(
    llm_service: LLMService, persona_details_str: str, year: int, month: int, day: int, daily_summary: str,
    monthly_summary: str, yearly_summary: str, location_for_day: Optional[str], current_year: int,
    current_month: int, current_day: int, current_hour: int
) -> Optional[Dict[int, Tuple[str, Optional[str]]]]:
    """Generates hourly breakdown with location."""
    logger.info(f"Generating hourly breakdown for {year}-{month:02d}-{day:02d} (Loc: {location_for_day})...")
    loc = location_for_day or 'Inferred Location'
    prompt = f"""
Persona: {persona_details_str} | Date: {year}-{month:02d}-{day:02d} | Location: **{loc}**
Context: Day="{daily_summary}", Month="{monthly_summary}", Year="{yearly_summary}"
**Task:** Generate detailed hour-by-hour breakdown (0-23).
**Instructions:** 1-2 sentences/hour. Include sleep, meals, hygiene, work/study/leisure, social, commute, breaks, chores. Consistent with context & location. Logical flow.
**Output Format:** ONLY valid JSON: {{"activities": [{{"hour": int, "location": str, "activity": str}}]}} (location always "{loc}"). All 24 hours.
"""
    validated_data = await _call_llm_and_get_validated_data( llm_service, prompt, HourlyBreakdownResponse, f"Hourly breakdown {year}-{month:02d}-{day:02d} ({loc})" )
    if validated_data and "activities" in validated_data:
        activities_list = validated_data["activities"]; activities_dict: Dict[int, Tuple[str, Optional[str]]] = {}; processed_hours = set(); valid = True
        for item_dict in activities_list:
            hour_num = item_dict.get("hour"); activity_text = item_dict.get("activity"); location_text = item_dict.get("location")
            if isinstance(hour_num, int) and 0 <= hour_num <= 23:
                if hour_num in processed_hours: logger.warning(f"Duplicate hour {hour_num}"); valid = False; break
                activities_dict[hour_num] = (activity_text, location_text); processed_hours.add(hour_num)
            else: logger.warning(f"Invalid hourly content: {item_dict}"); valid = False; break
        if not valid or not activities_dict: logger.warning(f"No valid hourly activities for {year}-{month:02d}-{day:02d}."); return None
        if year == current_year and month == current_month and day == current_day: # Pruning
            hours_to_remove = [h for h in activities_dict if h > current_hour]
            if hours_to_remove: logger.info(f"Pruning hours {hours_to_remove}");
            for h in hours_to_remove: del activities_dict[h]
        logger.info(f"Processed {len(activities_dict)} hourly activities for {year}-{month:02d}-{day:02d}."); return activities_dict
    else: logger.error(f"Failed validated hourly activities for {year}-{month:02d}-{day:02d}."); return None

# --- Orchestration Function moved here ---
async def generate_life_summary_sequentially(
    llm_service: LLMService, persona_details: Dict[str, Any], age: int
) -> Optional[Dict[str, Any]]:
    """Orchestrates life summary generation."""
    now = datetime.now(); today = now.date(); current_year = now.year; current_month = now.month; current_day = now.day; current_hour = now.hour; birth_year = current_year - age
    logger.info(f"Starting life summary orchestration (Age: {age}, Birth Year: {birth_year})...")
    life_summary = {"persona_details": persona_details, "age": age, "birth_year": birth_year, "birth_month": None, "birth_day": None, "initial_relationships": None, "yearly_summaries": {}, "monthly_summaries": {}, "daily_summaries": {}, "hourly_breakdowns": {}}
    details_list = [f"{k}: {v}" for k, v in persona_details.items()]; persona_details_str = ", ".join(details_list)
    ddgs_search_tool = DDGS()

    # Step 0: Initial Relationships
    initial_relationships_data = await generate_initial_relationships(llm_service, persona_details_str)
    initial_relationships_str = "No family data."
    if initial_relationships_data: life_summary["initial_relationships"] = initial_relationships_data; # Format str if needed
    else: logger.warning("Skipping relationship context due to generation failure.")

    # Step 1: Yearly Summaries & Birthday
    yearly_summaries: Dict[int, Tuple[str, Optional[str]]] = {}; birth_month, birth_day = None, None; total_summarized_years = 0
    yearly_result = await generate_yearly_summaries(llm_service, persona_details_str, initial_relationships_str, birth_year, current_year, current_year)
    if yearly_result: yearly_summaries, birth_month, birth_day = yearly_result; life_summary.update({"birth_month": birth_month, "birth_day": birth_day, "yearly_summaries": yearly_summaries}); total_summarized_years = len(yearly_summaries)
    else: logger.error("Failed yearly summaries. Aborting."); return life_summary

    # Step 2: Monthly Summaries
    years_for_monthly = [];
    if total_summarized_years > 0: available_years = sorted(yearly_summaries.keys()); monthly_count = min(math.ceil(total_summarized_years / 3), total_summarized_years); start_index = len(available_years) - monthly_count; years_for_monthly = available_years[start_index:] if len(available_years) >= monthly_count else available_years
    for year_to_process in years_for_monthly:
        if year_to_process in yearly_summaries: summary_text, location_text = yearly_summaries[year_to_process]; monthly_data = await generate_monthly_summaries_for_year(llm_service, persona_details_str, year_to_process, summary_text, location_text, current_year, current_month, ddgs_search_tool);
        if monthly_data: life_summary["monthly_summaries"][year_to_process] = monthly_data

    # Step 3: Daily Summaries (Plain Text)
    months_for_daily: List[Tuple[int, int]] = []; # Calculate current/previous month logic
    # ... (simplified calculation for brevity) ...
    months_for_daily.append((current_year, current_month)) # Example
    last_month_date = today.replace(day=1) - timedelta(days=1)
    months_for_daily.append((last_month_date.year, last_month_date.month)) # Example
    months_for_daily = sorted(list(set(months_for_daily))) # Unique & sorted

    for year_d, month_d in months_for_daily:
        yearly_data_tuple = yearly_summaries.get(year_d);
        if not yearly_data_tuple: continue
        yearly_context_daily, location_for_year = yearly_data_tuple
        monthly_data_tuple = life_summary["monthly_summaries"].get(year_d, {}).get(month_d); monthly_context_daily = monthly_data_tuple[0] if monthly_data_tuple else "N/A"
        daily_data = await generate_daily_summaries_for_month(llm_service, persona_details_str, year_d, month_d, monthly_context_daily, yearly_context_daily, location_for_year, current_year, current_month, current_day, ddgs_search_tool)
        if daily_data: life_summary["daily_summaries"].setdefault(year_d, {})[month_d] = daily_data

    # Step 4: Hourly Breakdowns
    days_for_hourly: List[Tuple[int, int, int]] = [];
    for i in range(7): target_date = today - timedelta(days=i); # Add birth date check if needed
    if target_date.year >= birth_year: days_for_hourly.append((target_date.year, target_date.month, target_date.day))
    days_for_hourly.reverse()
    for year, month, day in days_for_hourly:
         yearly_data_tuple = yearly_summaries.get(year); yearly_context_hr, location_for_year_hr = (yearly_data_tuple[0], yearly_data_tuple[1]) if yearly_data_tuple else ("N/A", None)
         monthly_data_tuple = life_summary["monthly_summaries"].get(year, {}).get(month); monthly_context_hr, location_for_month_hr = (monthly_data_tuple[0], monthly_data_tuple[1]) if monthly_data_tuple else ("N/A", location_for_year_hr)
         daily_summary_text = life_summary["daily_summaries"].get(year, {}).get(month, {}).get(day); location_for_day_hr = location_for_month_hr # Best guess
         if daily_summary_text is None: # On-demand daily
             logger.warning(f"Daily summary missing {year}-{month:02d}-{day:02d}. On-demand...")
             # ... (On-demand monthly/daily logic - simplified for brevity) ...
             daily_summary_text = f"On-demand summary for {day}" # Placeholder
         hourly_data = await generate_hourly_breakdown_for_day(llm_service, persona_details_str, year, month, day, daily_summary_text, monthly_context_hr, yearly_context_hr, location_for_day_hr, current_year, current_month, current_day, current_hour)
         if hourly_data: life_summary["hourly_breakdowns"].setdefault(year, {}).setdefault(month, {})[day] = hourly_data

    logger.info("Life summary orchestration finished.")
    return life_summary

# --- Top-Level Function moved here ---
async def generate_complete_persona_life(llm_service: LLMService) -> Optional[Dict[str, Any]]:
    """Generates a random persona and their full life summary."""
    logger.info("Starting full character life generation...")
    print("\n" + "*"*15 + " GENERATING RANDOM PERSONA " + "*"*15)
    generated_persona = await generate_random_persona(llm_service)
    if not generated_persona: logger.error("Persona generation failed."); return None
    print(f"Generated Persona:\n{json.dumps(generated_persona, indent=2)}")
    persona_age = generated_persona.get("Age")
    if not isinstance(persona_age, int) or persona_age < 1: logger.error(f"Invalid age '{persona_age}'."); return None
    print("\n" + "*"*15 + f" GENERATING LIFE SUMMARY (Age: {persona_age}) " + "*"*15)
    life_data = await generate_life_summary_sequentially(llm_service=llm_service, persona_details=generated_persona, age=persona_age)
    if not life_data: logger.error("Life summary generation failed."); return None
    logger.info("Full character life generation finished."); return life_data
