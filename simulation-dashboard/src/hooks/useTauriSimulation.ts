import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { SimulationState, EventData } from '../types/simulation';

interface CommandResponse {
  success: boolean;
  message: string;
  data?: any;
}

export const useTauriSimulation = () => {
  const [state, setState] = useState<SimulationState | null>(null);
  const [events, setEvents] = useState<EventData[]>([]);
  const [running, setRunning] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Extract events from simulation state  
  const extractEventsFromState = (stateData: SimulationState) => {
    const newEvents: EventData[] = [];

    console.log('🔍 Extracting events from state:', {
      narrative_log: stateData.narrative_log?.length || 0,
      simulacra_count: Object.keys(stateData.simulacra_profiles || {}).length,
      world_feeds: Object.keys(stateData.world_feeds || {}).length,
      recent_events: stateData.recent_events?.length || 0
    });

    // Use recent_events as the primary source since it has proper timestamps
    if (stateData.recent_events && Array.isArray(stateData.recent_events)) {
      console.log('🔍 Processing recent_events:', {
        total_events: stateData.recent_events.length,
        last_5_events: stateData.recent_events.slice(-5).map(e => `${e.event_type}(${e.agent_type})`)
      });
      
      stateData.recent_events.slice(-50).forEach((event: any) => {
        // Use simulation time as timestamp to avoid Date.now() creating new timestamps
        const eventTimestamp = event.sim_time_s || stateData.world_time || 0;
        
        // Simulacra monologue events from recent_events
        if (event.event_type === 'monologue' && event.agent_type === 'simulacra') {
          console.log('✅ Found Simulacra monologue event:', event.sim_time_s);
          const agentName = event.agent_id;
          newEvents.push({
            timestamp: eventTimestamp,
            sim_time_s: event.sim_time_s || 0,
            agent_id: agentName,
            event_type: 'monologue',
            data: { 
              content: event.data.monologue || event.data.content || 'Agent thinking...',
              monologue: event.data.monologue || event.data.content || 'Agent thinking...'
            }
          });
        }
        
        // Simulacra observation events
        if (event.event_type === 'observation' && event.agent_type === 'simulacra') {
          const agentName = event.agent_id;
          newEvents.push({
            timestamp: eventTimestamp,
            sim_time_s: event.sim_time_s || 0,
            agent_id: agentName,
            event_type: 'observation',
            data: { 
              content: event.data.observation || event.data.content || 'Agent observing...',
              observation: event.data.observation || event.data.content || 'Agent observing...'
            }
          });
        }
        
        // Narrative events from recent_events
        if (event.event_type === 'narrative' && (event.agent_type === 'narrator' || event.agent_id === 'Narrator')) {
          console.log('✅ Found Narrative event:', event.sim_time_s);
          newEvents.push({
            timestamp: eventTimestamp,
            sim_time_s: event.sim_time_s || 0,
            agent_id: 'Narrator',
            event_type: 'narrative',
            data: { 
              content: event.data.narrative_text || event.data.content || 'Narrative update...',
              narrative_text: event.data.narrative_text || event.data.content || 'Narrative update...'
            }
          });
        }

        // World Engine resolution events
        if ((event.event_type === 'resolution' || event.event_type === 'world_engine') && event.agent_type === 'world_engine') {
          console.log('✅ Found World Engine event:', event.sim_time_s);
          const resolutionData = event.data || {};
          const content = resolutionData.outcome_description || 'World Engine resolution';
          const details = JSON.stringify({
            valid_action: resolutionData.valid_action,
            duration: resolutionData.duration,
            results: resolutionData.results,
            scheduled_future_event: resolutionData.scheduled_future_event
          }, null, 2);
          
          newEvents.push({
            timestamp: eventTimestamp,
            sim_time_s: event.sim_time_s || 0,
            agent_id: 'World Engine',
            event_type: 'world_engine',
            data: { 
              content: content,
              details: details,
              resolution: resolutionData
            }
          });
        }
      });
    }
    
    // Extract narrative from narrative_log as fallback
    if (stateData.narrative_log && Array.isArray(stateData.narrative_log) && stateData.narrative_log.length > 0) {
      console.log('📖 Extracting narrative from narrative_log:', stateData.narrative_log.length);
      
      // Get the last 10 narrative entries
      stateData.narrative_log.slice(-10).forEach((narrative: string, index: number) => {
        // Extract the timestamp from narrative text like "[T56.7]"
        const timestampMatch = narrative.match(/\[T([\d.]+)\]/);
        const narrativeTimestamp = timestampMatch ? parseFloat(timestampMatch[1]) : (stateData.world_time || 0) - (index * 0.1);
        
        newEvents.push({
          timestamp: narrativeTimestamp,
          sim_time_s: narrativeTimestamp,
          agent_id: 'Narrator',
          event_type: 'narrative',
          data: { 
            content: narrative,
            narrative_text: narrative
          }
        });
      });
    }
    
    // Fallback: Extract basic World Engine activity from state
    if (stateData.current_world_state && newEvents.filter(e => e.event_type === 'world_engine').length === 0) {
      const worldEngineActivity = `World time: ${(stateData.world_time || 0).toFixed(1)}s`;
      newEvents.push({
        timestamp: stateData.world_time || 0,
        sim_time_s: stateData.world_time || 0,
        agent_id: 'World Engine',
        event_type: 'world_engine',
        data: { 
          content: worldEngineActivity,
          details: 'Processing simulation state'
        }
      });
    }

    // Extract world feeds as events
    if (stateData.world_feeds) {
      if (stateData.world_feeds.weather) {
        newEvents.push({
          timestamp: (stateData.world_time || 0) - 0.1,
          sim_time_s: stateData.world_time || 0,
          agent_id: 'World Engine',
          event_type: 'world_engine',
          data: { 
            content: `Weather: ${stateData.world_feeds.weather.condition}`,
            details: `Temperature: ${stateData.world_feeds.weather.temperature_celsius}°C`
          }
        });
      }
    }

    // Extract agent events from simulacra profiles
    if (stateData.simulacra_profiles) {
      Object.entries(stateData.simulacra_profiles as any).forEach(([agentId, agentData]: [string, any]) => {
        const agentName = agentData.persona_details?.Name || agentId;

        // Extract monologue events from monologue_history
        if (agentData.monologue_history && Array.isArray(agentData.monologue_history)) {
          console.log(`✅ Found ${agentData.monologue_history.length} monologue entries for ${agentName}`);
          
          agentData.monologue_history.slice(-10).reverse().forEach((monologue: string, index: number) => {
            // Extract timestamp from monologue if it has one like "[T114.5]"
            const timestampMatch = monologue.match(/\[T([0-9.]+)\]/);
            const timestamp = timestampMatch ? parseFloat(timestampMatch[1]) : (stateData.world_time || 0) - (index * 0.1);
            
            newEvents.push({
              timestamp: timestamp,
              sim_time_s: timestamp,
              agent_id: agentName,
              event_type: 'monologue',
              data: { 
                content: monologue,
                monologue: monologue
              }
            });
          });
        }

        // Extract current action as intent event
        if (agentData.current_action_description) {
          newEvents.push({
            timestamp: stateData.world_time || 0,
            sim_time_s: stateData.world_time || 0,
            agent_id: agentName,
            event_type: 'intent',
            data: { 
              action_type: agentData.current_action_description.split(' - ')[0]?.replace('Action: ', '') || 'unknown',
              details: agentData.current_action_description.split(' - ')[1] || agentData.current_action_description,
              content: `Status: ${agentData.status} | Action: ${agentData.current_action_description}`,
              status: agentData.status
            }
          });
        }

        // Extract last observation as observation event
        if (agentData.last_observation) {
          newEvents.push({
            timestamp: (stateData.world_time || 0) - 0.1,
            sim_time_s: stateData.world_time || 0,
            agent_id: agentName,
            event_type: 'observation',
            data: { 
              content: agentData.last_observation,
              observation: agentData.last_observation
            }
          });
        }
      });
    }

    // Always update events, even if empty, and increase limit
    const sortedEvents = newEvents.sort((a, b) => b.timestamp - a.timestamp).slice(0, 50);
    setEvents(sortedEvents);
  };

  // Get current simulation state
  const refreshState = useCallback(async () => {
    try {
      const result = await invoke<SimulationState | null>('get_simulation_state');
      if (result) {
        setState(result);
        extractEventsFromState(result);
        setError(null);
      }
    } catch (err) {
      console.error('Failed to get simulation state:', err);
      setError(err as string);
    }
  }, []);

  // Start simulation
  const startSimulation = useCallback(async () => {
    try {
      const result = await invoke<CommandResponse>('start_simulation');
      if (result.success) {
        setRunning(true);
        setError(null);
      } else {
        setError(result.message);
      }
      return result;
    } catch (err) {
      const errorMsg = err as string;
      setError(errorMsg);
      return { success: false, message: errorMsg };
    }
  }, []);

  // Stop simulation
  const stopSimulation = useCallback(async () => {
    try {
      const result = await invoke<CommandResponse>('stop_simulation');
      if (result.success) {
        setRunning(false);
      }
      return result;
    } catch (err) {
      const errorMsg = err as string;
      setError(errorMsg);
      return { success: false, message: errorMsg };
    }
  }, []);

  // Command functions
  const injectNarrative = useCallback(async (text: string) => {
    try {
      return await invoke<CommandResponse>('inject_narrative', { text });
    } catch (err) {
      return { success: false, message: err as string };
    }
  }, []);

  const sendAgentEvent = useCallback(async (agentId: string, description: string) => {
    try {
      return await invoke<CommandResponse>('send_agent_event', { 
        agent_id: agentId, 
        description 
      });
    } catch (err) {
      return { success: false, message: err as string };
    }
  }, []);

  const updateWorldInfo = useCallback(async (category: string, info: string) => {
    try {
      return await invoke<CommandResponse>('update_world_info', { 
        category, 
        info 
      });
    } catch (err) {
      return { success: false, message: err as string };
    }
  }, []);

  const teleportAgent = useCallback(async (agentId: string, location: string) => {
    try {
      return await invoke<CommandResponse>('teleport_agent', { 
        agent_id: agentId, 
        location 
      });
    } catch (err) {
      return { success: false, message: err as string };
    }
  }, []);

  const interactiveChat = useCallback(async (agentId: string, message: string) => {
    try {
      return await invoke<CommandResponse>('interactive_chat', { 
        agent_id: agentId, 
        message 
      });
    } catch (err) {
      return { success: false, message: err as string };
    }
  }, []);

  // Simple input focus management - NO RE-RENDERS
  const setInputFocus = useCallback((active: boolean) => {
    // This is just a placeholder - we'll handle focus differently
    console.log('Input focus:', active);
  }, []);

  // Setup event listeners - SIMPLE VERSION
  useEffect(() => {
    let unlistenStateUpdate: (() => void) | undefined;

    const setup = async () => {
      try {
        // Listen for state updates from Rust backend
        unlistenStateUpdate = await listen<SimulationState>('simulation-state-update', (event) => {
          console.log('🔄 Received state update from Tauri:', {
            world_time: event.payload.world_time,
            active_simulacra: event.payload.active_simulacra_ids?.length || 0
          });
          setState(event.payload);
          extractEventsFromState(event.payload);
        });

        // Start Redis connection
        console.log('🚀 Starting Redis connection...');
        const startResult = await startSimulation();
        if (startResult.success) {
          console.log('✅ Redis connection started successfully');
          setRunning(true);
        } else {
          console.error('❌ Failed to start Redis connection:', startResult.message);
          setError(startResult.message);
        }

        // Get initial state
        await refreshState();
        
        // Remove polling - rely only on Redis event-driven updates
        console.log('📡 Relying on Redis event-driven updates only');
        
        setLoading(false);
      } catch (err) {
        console.error('❌ Failed to setup simulation:', err);
        setError(err as string);
        setLoading(false);
      }
    };

    setup();

    return () => {
      if (unlistenStateUpdate) {
        unlistenStateUpdate();
      }
    };
  }, []); // Empty dependency array

  return {
    // State
    state,
    events,
    loading,
    error,
    running,
    
    // Actions
    startSimulation,
    stopSimulation,
    refreshState,
    
    // Commands
    injectNarrative,
    sendAgentEvent,
    updateWorldInfo,
    teleportAgent,
    interactiveChat,
    
    // Input management
    setInputFocus
  };
};