import React, { createContext, useContext, useRef, useCallback, useState, useEffect, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { SimulationState, EventData } from '../types/simulation';

interface CommandResponse {
  success: boolean;
  message: string;
  data?: any;
}

interface SimulationContextType {
  state: SimulationState | null;
  events: EventData[];
  loading: boolean;
  error: string | null;
  running: boolean;
  
  // Actions
  startSimulation: () => Promise<CommandResponse>;
  stopSimulation: () => Promise<CommandResponse>;
  refreshState: () => Promise<void>;
  
  // Commands
  injectNarrative: (text: string) => Promise<CommandResponse>;
  sendAgentEvent: (agentId: string, description: string) => Promise<CommandResponse>;
  updateWorldInfo: (category: string, info: string) => Promise<CommandResponse>;
  teleportAgent: (agentId: string, location: string) => Promise<CommandResponse>;
  interactiveChat: (agentId: string, message: string) => Promise<CommandResponse>;
}

const SimulationContext = createContext<SimulationContextType | undefined>(undefined);

export const useSimulation = () => {
  const context = useContext(SimulationContext);
  if (!context) {
    throw new Error('useSimulation must be used within a SimulationProvider');
  }
  return context;
};

export const SimulationProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [state, setState] = useState<SimulationState | null>(null);
  const [events, setEvents] = useState<EventData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  
  

  // Extract events from simulation state
  const extractEventsFromState = useCallback((stateData: SimulationState) => {
    const newEvents: EventData[] = [];
    const currentTime = Date.now() / 1000;

    // Extract narrative events
    if (stateData.narrative_log && Array.isArray(stateData.narrative_log)) {
      stateData.narrative_log.slice(0, 3).forEach((narrative: string, index: number) => {
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

    // Extract agent events
    if (stateData.simulacra_profiles) {
      Object.entries(stateData.simulacra_profiles as any).forEach(([agentId, agentData]: [string, any]) => {
        if (agentData.monologue_history && Array.isArray(agentData.monologue_history)) {
          agentData.monologue_history.slice(0, 3).forEach((monologue: string, index: number) => {
            newEvents.push({
              timestamp: currentTime - index,
              sim_time_s: stateData.world_time || 0,
              agent_id: agentData.persona_details?.Name || agentId,
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
            timestamp: currentTime,
            sim_time_s: stateData.world_time || 0,
            agent_id: agentData.persona_details?.Name || agentId,
            event_type: 'intent',
            data: { 
              action_type: agentData.current_action_description.split(' - ')[0]?.replace('Action: ', '') || 'unknown',
              details: agentData.current_action_description.split(' - ')[1] || agentData.current_action_description,
              content: agentData.current_action_description
            }
          });
        }
      });
    }

    if (newEvents.length > 0) {
      setEvents(prev => [...newEvents.reverse(), ...prev.slice(0, 200 - newEvents.length)]);
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
      return { success: true, message: `World info updated: ${category}` };
    } catch (err) {
      return { success: false, message: err as string };
    }
  }, []);

  const teleportAgent = useCallback(async (agentId: string, location: string) => {
    try {
      return { success: true, message: `Agent ${agentId} teleported to ${location}` };
    } catch (err) {
      return { success: false, message: err as string };
    }
  }, []);

  const interactiveChat = useCallback(async (agentId: string, message: string) => {
    try {
      return { success: true, message: `Message sent to ${agentId}` };
    } catch (err) {
      return { success: false, message: err as string };
    }
  }, []);

  

  // Setup event listeners - ONCE
  useEffect(() => {
    let unlistenStateUpdate: (() => void) | undefined;

    const setup = async () => {
      try {
        // Listen for state updates from Rust backend
        unlistenStateUpdate = await listen<SimulationState>('simulation-state-update', (event) => {
          console.log('Received state update from Tauri:', event.payload);
          if (true) {
            setState(event.payload);
            extractEventsFromState(event.payload);
          }
        });

        // Start Redis connection
        const startResult = await startSimulation();
        if (startResult.success) {
          console.log('Redis connection started successfully');
        } else {
          console.error('Failed to start Redis connection:', startResult.message);
          setError(startResult.message);
        }

        // Get initial state
        await refreshState();
        setLoading(false);
      } catch (err) {
        console.error('Failed to setup simulation:', err);
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

  const value = useMemo(() => ({
    state,
    events,
    loading,
    error,
    running,
    startSimulation,
    stopSimulation,
    refreshState,
    injectNarrative,
    sendAgentEvent,
    updateWorldInfo,
    teleportAgent,
    interactiveChat
  }), [
    state, 
    events, 
    loading, 
    error, 
    running, 
    startSimulation, 
    stopSimulation, 
    refreshState, 
    injectNarrative, 
    sendAgentEvent, 
    updateWorldInfo, 
    teleportAgent, 
    interactiveChat
  ]);

  return (
    <SimulationContext.Provider value={value}>
      {children}
    </SimulationContext.Provider>
  );
};