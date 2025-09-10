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
  const extractEventsFromState = useCallback((stateData: SimulationState) => {
    const newEvents: EventData[] = [];
    const currentTime = Date.now() / 1000;

    console.log('🔍 Extracting events from state:', {
      narrative_log: stateData.narrative_log?.length || 0,
      simulacra_count: Object.keys(stateData.simulacra_profiles || {}).length,
      world_feeds: Object.keys(stateData.world_feeds || {}).length
    });

    // Extract narrative events from narrative_log
    if (stateData.narrative_log && Array.isArray(stateData.narrative_log)) {
      stateData.narrative_log.slice(-10).reverse().forEach((narrative: string, index: number) => {
        newEvents.push({
          timestamp: currentTime - index,
          sim_time_s: stateData.world_time || 0,
          agent_id: 'Narrator',
          event_type: 'narrative',
          data: { 
            content: narrative,
            narrative_text: narrative
          }
        });
      });
    }

    // Extract events from recent_events log
    if (stateData.recent_events && Array.isArray(stateData.recent_events)) {
      console.log('🔍 Processing recent_events:', {
        total_events: stateData.recent_events.length,
        last_5_events: stateData.recent_events.slice(-5).map(e => `${e.event_type}(${e.agent_type})`)
      });
      
      // Also set window title for debugging
      if (typeof window !== 'undefined') {
        window.document.title = `Dashboard - ${stateData.recent_events.length} events`;
      }
      
      stateData.recent_events.slice(-20).forEach((event: any) => {
        // World Engine resolution events
        if (event.event_type === 'resolution' && event.agent_type === 'world_engine') {
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
            timestamp: currentTime - ((stateData.world_time || 0) - (event.sim_time_s || 0)),
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
        
        // Simulacra observation events
        if (event.event_type === 'observation' && event.agent_type === 'simulacra') {
          const agentName = event.agent_id;
          newEvents.push({
            timestamp: currentTime - ((stateData.world_time || 0) - (event.sim_time_s || 0)),
            sim_time_s: event.sim_time_s || 0,
            agent_id: agentName,
            event_type: 'observation',
            data: { 
              content: event.data.observation || event.data.content || 'Agent observing...',
              observation: event.data.observation || event.data.content || 'Agent observing...'
            }
          });
        }
        
        // Simulacra monologue events from recent_events
        if (event.event_type === 'monologue' && event.agent_type === 'simulacra') {
          console.log('✅ Found Simulacra monologue event:', event.sim_time_s);
          const agentName = event.agent_id;
          newEvents.push({
            timestamp: currentTime - ((stateData.world_time || 0) - (event.sim_time_s || 0)),
            sim_time_s: event.sim_time_s || 0,
            agent_id: agentName,
            event_type: 'monologue',
            data: { 
              content: event.data.monologue || event.data.content || 'Agent thinking...',
              monologue: event.data.monologue || event.data.content || 'Agent thinking...'
            }
          });
        }
      });
    }
    
    // Fallback: Extract basic World Engine activity from state
    if (stateData.current_world_state && newEvents.filter(e => e.event_type === 'world_engine').length === 0) {
      const worldEngineActivity = `World time: ${(stateData.world_time || 0).toFixed(1)}s`;
      newEvents.push({
        timestamp: currentTime,
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
          timestamp: currentTime - 1,
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

    // Extract agent events with better handling
    if (stateData.simulacra_profiles) {
      Object.entries(stateData.simulacra_profiles as any).forEach(([agentId, agentData]: [string, any]) => {
        const agentName = agentData.persona_details?.Name || agentId;

        // Extract recent monologues (last 10) - full content for simulacra tab
        if (agentData.monologue_history && Array.isArray(agentData.monologue_history)) {
          agentData.monologue_history.slice(-10).reverse().forEach((monologue: string, index: number) => {
            newEvents.push({
              timestamp: currentTime - index - 2,
              sim_time_s: stateData.world_time || 0,
              agent_id: agentName,
              event_type: 'monologue',
              data: { 
                content: monologue, // Full monologue content
                monologue: monologue
              }
            });
          });
        }

        // Extract current action status
        if (agentData.current_action_description) {
          newEvents.push({
            timestamp: currentTime - 0.5,
            sim_time_s: stateData.world_time || 0,
            agent_id: agentName,
            event_type: 'intent',
            data: { 
              action_type: agentData.current_action_description.split(' - ')[0]?.replace('Action: ', '') || 'unknown',
              details: agentData.current_action_description.split(' - ')[1] || agentData.current_action_description,
              content: `${agentData.status}: ${agentData.current_action_description}`,
              status: agentData.status
            }
          });
        }

        // Extract recent observations (full content)
        if (agentData.last_observation) {
          newEvents.push({
            timestamp: currentTime - 1.5,
            sim_time_s: stateData.world_time || 0,
            agent_id: agentName,
            event_type: 'observation',
            data: { 
              content: agentData.last_observation, // Full observation content
              observation: agentData.last_observation
            }
          });
        }
      });
    }

    // Limit to last 10 events and update
    if (newEvents.length > 0) {
      const sortedEvents = newEvents.sort((a, b) => b.timestamp - a.timestamp).slice(0, 10);
      setEvents(sortedEvents);
    }
  }, []);

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
  }, [extractEventsFromState]);

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