export interface SimulationState {
  world_time: number;
  world_instance_uuid: string;
  simulacra_profiles: Record<string, SimulacraProfile>;
  active_simulacra_ids: string[];
  current_world_state: {
    location_details: Record<string, LocationDetails>;
  };
  world_feeds: WorldFeeds;
  world_template_details: {
    description: string;
    name: string;
  };
}

export interface SimulacraProfile {
  persona_details: {
    Name: string;
    Age: number;
    Background: string;
    Personality: string;
    Goals: string;
  };
  status: 'idle' | 'busy' | 'thinking';
  current_location: string;
  current_action_description: string;
  current_action_end_time: number;
}

export interface LocationDetails {
  name: string;
  description: string;
  ephemeral_objects: EphemeralObject[];
  ephemeral_npcs: EphemeralNPC[];
  connected_locations: Connection[];
}

export interface EphemeralObject {
  name: string;
  description: string;
  type: string;
}

export interface EphemeralNPC {
  name: string;
  description: string;
  personality: string;
}

export interface Connection {
  to_location_id_hint: string;
  description: string;
  type: string;
}

export interface WorldFeeds {
  weather: {
    condition: string;
    temperature: number;
    description: string;
  };
  local_news: {
    headlines: string[];
  };
  world_news: {
    headlines: string[];
  };
  regional_news: {
    headlines: string[];
  };
}

export interface EventData {
  timestamp: number;
  sim_time_s: number;
  agent_id: string;
  event_type: 'monologue' | 'intent' | 'resolution' | 'narrative';
  data: any;
}

export interface MonologueEvent {
  monologue: string;
}

export interface IntentEvent {
  action_type: string;
  details: string;
  target_id: string;
}

export interface ResolutionEvent {
  valid_action: boolean;
  duration: number;
  outcome_description: string;
  results: {
    discovered_objects: EphemeralObject[];
    discovered_connections: Connection[];
  };
}

export interface NarrativeEvent {
  narrative_text: string;
}