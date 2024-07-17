from enum import Enum
import json
import yaml

### TBD integration with diffusion model for image generation.
### Interaction mechanism with simulated world or subjects.
class prompts(Enum):
    with open('config.yaml', 'r') as file:
        configa = yaml.safe_load(file)

    prompt_non_openai = "\nRespond in the following JSON format: " + json.dumps(
            {
            "time_tick": "Time movement",
            "world_delta": "Update world changes",
            "narration_delta": "",
            "self_consistancy_check": "Reason output for internal consistency.",
            "subject_thought_process": "Internal State of the simulacrum",
            "actions": "",
            "state_description": "Overall state description/summary - send to ImageGen to visualize the current state",
            }
    )

    prompt_world_engine =f"""Instructions:
world simulation prompt placeholder

World Interaction mechanism to affect the physical state of the world.

World Engine Instructions:

Setting the Stage:

Upon initialization, establish the fundamental parameters of the world:
a) Time period (historical, present, future, or alternate timeline)
b) Physical location (Earth, another planet, fictional world, etc.)
c) Technological level
d) Presence of magic or supernatural elements (if any)
e) Major cultural and societal norms

Physical Laws:
Enforce the known laws of physics as understood in our reality, unless otherwise specified.
If custom physical laws are provided in the prompt, document and consistently apply these throughout all interactions.

Consistency Maintenance:
Keep a running log of all established facts, events, and character actions.
Ensure all subsequent narrations and interactions align with previously established information.

Environmental Factors:
Simulate appropriate environmental conditions based on the setting (weather, day/night cycle, seasons, etc.).
Account for how these conditions affect characters and events.

Cause and Effect:
Model realistic consequences for all actions and events within the world.
Consider both immediate and long-term effects of significant occurrences.

Time Management:
Keep track of the passage of time within the world.
Ensure events unfold in a logically consistent timeline.

Spatial Awareness:
Maintain a coherent understanding of locations and distances within the world.
Enforce realistic travel times and spatial relationships.

Character Interactions:
Simulate realistic behaviors for non-player characters (NPCs) based on their established traits and the world's rules.
Ensure character actions are consistent with their abilities and the world's physical laws.

Resource Management:
Track the availability and consumption of resources relevant to the world and its inhabitants.
Implement realistic resource constraints and their effects on the world.

Conflict Resolution:
Provide impartial outcomes for conflicts based on established rules, character abilities, and random chance where appropriate.
Ensure that the resolution of conflicts adheres to the world's physical and social laws.

Adaptation:
Be prepared to incorporate new elements introduced by users or other sources, ensuring they fit coherently into the existing world structure.
Adjust the world state as necessary in response to significant events or user actions.

Narrative Cohesion:
Maintain a cohesive narrative structure while allowing for dynamic, user-driven changes to the world.
Provide background information and context as needed to support immersion and understanding.

Limitations and Boundaries:
Clearly communicate the boundaries and limitations of the world to users.
If a requested action is impossible within the established rules, explain why and suggest alternatives if applicable.

Documentation:
Maintain detailed records of all significant world elements, events, and changes.
Be prepared to provide this information to users upon request for clarity and continuity.

Fundemental Time Units:
Maintain Ticks as in action time units as follows - 1 minute, 1 hour, 1 day, 1 week, 1 month, 1 year
Depending on the tick context, world interactions must follow these fundemental unit of time.

Remember, as the world engine, your primary function is to create and maintain a consistent, immersive, and logically coherent environment for narrative and interaction. Always prioritize the integrity of the established world rules while facilitating engaging and dynamic experiences within those constraints.

"""
    prompt_narration_engine =f"""Instructions:
narration - engine prompt place holder ### Function calls in JSON return format - GROQ, Anthropic, Local LLM needs to follow RETURN IN JSON instruction.
# OpenAI - Function Call returns

Generate a detailed human biography following this structure:

Basic Information:
Full name:
Current age
Place of birth
Current location

Life Overview:
Divide the person's life into thirds based on their current age. For each third, provide the following:
First Third of Life (Birth to [Age/3]):

Provide yearly highlights and significant events
Include major developmental milestones, family dynamics, and educational experiences

Second Third of Life ([Age/3] to [2*Age/3]):
Break this period down month by month
Detail important life events, career developments, relationships, and personal growth

Final Third of Life ([2*Age/3] to present):
Divide this last third into three parts:
a) First part: Provide weekly summaries

Focus on career advancements, family life, and significant personal events
b) Second part: Give daily accounts

Describe routines, daily challenges, and small but meaningful occurrences
c) Final part (most recent): Provide an hourly breakdown of a typical day

Detail the person's current daily life, including work, leisure, and personal time

Conclusion:
Summarize the person's current state of life
Mention their hopes, dreams, and plans for the future

Throughout the biography, include details about:
Personality traits and how they evolved
Key relationships and their impact
Major life decisions and their consequences
Challenges faced and overcome
Achievements and disappointments
Changes in worldview or personal philosophy

Fundemental Time Units:
Maintain Ticks as in action time units as follows - 1 minute, 1 hour, 1 day, 1 week, 1 month, 1 year
Depending on the tick context, world interactions must follow these fundemental unit of time.

Ensure the biography feels cohesive and authentic, with each part of life naturally leading into the next. The level of detail should increase as the timeline approaches the present day.
"""
    boltzmann_brain_prompt =f"""Instructions:
Prompt Place Holder:

Sample Prompt: #Backstory template needs to generate a full profile of individual. To be generated by another LLM.

City and country: [Specify]
Neighborhood: [Type and characteristics]
Living situation: [House, apartment, shared living, etc.]
Notable local landmarks or features: [Nearby points of interest]

Identity:

Name: [Full name, including any nicknames]
Age: [Exact age]
Gender: [Gender identity and preferred pronouns]
Occupation: [Job title, company/organization, brief description of role]
Ethnicity: [Ethnic background and cultural identity]
Nationality: [Country of citizenship and any dual citizenships]
Languages spoken: [List with proficiency levels]

Physical Appearance:

Height and build: [Specific details]
Hair and eye color: [Description]
Distinctive features: [Any notable physical characteristics]
Style of dress: [Typical clothing choices and any signature items]

Current Time: [Specific date, day of the week, and time of day]
Current Activity:

Primary action: [What they're doing right now]
Location of activity: [Where this is taking place]
Purpose: [Why they're engaged in this activity]
Emotional state: [How they're feeling about it]

Backstory:

Childhood: [Brief overview of upbringing and significant childhood experiences]
Family background: [Parents' occupations, siblings, family dynamics]
Education: [Complete educational history, including any specialized training]
Career path: [Overview of professional journey]
Key life events: [3-5 significant experiences that shaped the person]
Relationship history: [Brief overview of past and current romantic relationships]
Places lived: [List of locations they've called home]

Personality Traits:

[List 5-7 defining personality characteristics with brief explanations]
Myers-Briggs Type: [If applicable]
Strengths: [3-4 positive attributes]
Weaknesses: [3-4 areas for personal growth]

Beliefs and Values:

Political leaning: [Detailed description of political views and involvement]
Religious/spiritual beliefs: [Specific denomination or philosophy, level of devotion]
Core values: [List 3-5 fundamental principles the person lives by]
Ethical framework: [How they make moral decisions]
Worldview: [General outlook on life and humanity]

Interests and Hobbies:

[List 5-7 activities or subjects the person is passionate about]
Skills: [Any special abilities or talents related to their interests]
Collections: [Any items they collect as part of their hobbies]

Goals and Aspirations:

Short-term goals: [2-3 objectives for the near future]
Long-term goals: [2-3 major life objectives]
Dream scenario: [Their ideal life situation]
Career ambitions: [Professional aspirations]

Challenges:

Current problems: [2-3 issues the person is dealing with right now]
Ongoing struggles: [2-3 persistent difficulties in their life]
Fears and anxieties: [What keeps them up at night]
Regrets: [Past decisions or actions they wish they could change]

Communication Style:

Speech patterns: [Distinctive ways of speaking, accent, vocabulary choices]
Body language: [Typical non-verbal cues and gestures]
Conflict resolution style: [How they handle disagreements]
Emotional expression: [How they show or hide their feelings]

Social Circle:

Family relationships: [Current dynamics with immediate and extended family]
Close friends: [Brief description of 3-5 important friendships]
Professional network: [Key connections in their career field]
Community involvement: [Any groups or organizations they're part of]
Social media presence: [Platforms used, frequency of engagement, type of content shared]

Daily Routine:

Weekday schedule: [Detailed hour-by-hour breakdown]
Weekend activities: [Typical leisure time pursuits]
Eating habits: [Dietary preferences, favorite foods, meal routines]
Exercise regimen: [Any regular physical activities]
Sleep patterns: [Typical sleep schedule and any sleep-related issues]

Recent Experiences:

[3-4 notable events from the past month that are on their mind]
Current projects: [Any ongoing personal or professional endeavors]
Latest accomplishment: [A recent achievement they're proud of]
Recent challenge: [A difficult situation they've faced lately]

Financial Situation:

Income level: [General range and sources of income]
Spending habits: [How they manage their money]
Savings and investments: [Financial goals and strategies]
Debts: [Any significant financial obligations]

Health and Wellness:

Overall health status: [Any chronic conditions or health concerns]
Mental health: [Any diagnosed conditions or ongoing therapy]
Self-care practices: [How they maintain their well-being]
Relationship with healthcare: [Frequency of check-ups, attitudes toward medicine]

Cultural Touchstones:

Favorite media: [Books, movies, TV shows, music they enjoy]
Influential figures: [People they admire or who have impacted their life]
Cultural identity: [How they relate to their heritage and current cultural environment]

Environment and Possessions:

Living space: [Description of home interior and any prized possessions]
Transportation: [How they typically get around]
Technology: [Devices they use and their comfort level with tech]
Pets: [Any animal companions and their significance]

"""

    functions = [
    {
        "name": "RUN",
        "description": "description",
        "parameters": {
            "type": "object",
            "properties": {
                "time_tick": {
                    "type": "string",
                    "description": ".",
                },
                "world_state": {
                    "type": "string",
                    "description": "Think Step by Step and specify the trading action to be taken. This decision should be supported by ALL the data provided.",
                },
                "world_delta": {
                    "type": "string",
                    "description": "Think Step by Step and Provide an in-depth analysis and justification considering all available data for your chosen action. REMEMBER where the current price is in respect to the MEAN, MARKET HIGH and LOW. Elaborate on why alternative trading actions were deemed unsuitable.",
                },
                "narration_delta": {
                    "type": "string",
                    "description": "Encourage constructive criticism and self-evaluation. Describe what information is not helpful, especially examine the options chain and their greeks. For DO_NOTHING actions, discuss the rationale and describe what the ideal conditions for trading would be.",
                },
                "self_consistancy_check": {
                    "type": "string",
                    "description": "List known trading constraints and any additional constraints extrapolated from current market conditions, explaining how these influence trading decisions.",
                },
                "subject_thought_process": {
                    "type": "string",
                    "description": ".",             
                },
                "actions": {
                    "type": "string",
                    "description": ".",
                },
                "stats_trend": {
                    "type": "string",
                    "description": "Enum example.",
                    "enum": [
                            "A",
                            "B",
                            "C",
                            ],
                },
                "state_description": {
                    "type": "string",
                    "description": "Determine if the market is FLAT, SIDEWAYS, RANGE-BOUND, or CONSOLIDATING based on historical price data and all technical indicators provided.",
                },
            },
            "required": ["time_tick", "world_delta", "narration_delta", "self_consistancy_check", "subject_thought_process", "actions", "state_description"],
        },
    },
]