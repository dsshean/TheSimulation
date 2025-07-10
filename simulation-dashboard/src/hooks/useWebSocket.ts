import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import { SimulationState, EventData } from '../types/simulation';

interface WebSocketMessage {
  type: 'state_update' | 'event' | 'command_response';
  timestamp: string;
  data: any;
  request_id?: string;
  response?: any;
}

interface CommandResponse {
  success: boolean;
  message: string;
  data?: any;
}

export const useWebSocket = () => {
  // Separate state pieces to allow targeted updates
  const [worldState, setWorldState] = useState<any>(null);
  const [simulacraProfiles, setSimulacraProfiles] = useState<any>({});
  const [activeSimulacraIds, setActiveSimulacraIds] = useState<string[]>([]);
  const [worldTime, setWorldTime] = useState<number>(0);
  const [worldFeeds, setWorldFeeds] = useState<any>({});
  const [narrativeLog, setNarrativeLog] = useState<string[]>([]);
  const [events, setEvents] = useState<EventData[]>([]);
  
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>();
  const pendingCommands = useRef<Map<string, (response: CommandResponse) => void>>(new Map());
  
  // Combine individual state pieces into full state for compatibility
  const state = useMemo(() => {
    if (!worldState && !simulacraProfiles) return null;
    
    return {
      world_time: worldTime,
      world_instance_uuid: worldState?.world_instance_uuid || '',
      simulacra_profiles: simulacraProfiles,
      active_simulacra_ids: activeSimulacraIds,
      current_world_state: worldState?.current_world_state || {},
      world_feeds: worldFeeds,
      world_template_details: worldState?.world_template_details || {},
      narrative_log: narrativeLog
    };
  }, [worldState, simulacraProfiles, activeSimulacraIds, worldTime, worldFeeds, narrativeLog]);

  const lastExtractedState = useRef<any>(null);
  
  const extractEventsFromState = useCallback((stateData: any) => {
    // Only extract events if state has actually changed
    if (JSON.stringify(stateData) === JSON.stringify(lastExtractedState.current)) {
      return;
    }
    lastExtractedState.current = stateData;
    
    const newEvents: EventData[] = [];
    const currentTime = Date.now() / 1000;

    // Extract narrative events from narrative_log
    if (stateData.narrative_log && Array.isArray(stateData.narrative_log)) {
      stateData.narrative_log.slice(0, 3).forEach((narrative: string, index: number) => {
        newEvents.push({
          timestamp: currentTime - index, // Slightly different timestamps
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

    // Extract agent monologue events
    if (stateData.simulacra_profiles) {
      Object.entries(stateData.simulacra_profiles).forEach(([agentId, agentData]: [string, any]) => {
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

    // Add new events to the beginning of the events array
    if (newEvents.length > 0) {
      setEvents(prev => [...newEvents.reverse(), ...prev.slice(0, 200 - newEvents.length)]);
    }
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    try {
      const ws = new WebSocket('ws://localhost:8766');
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        setConnected(true);
        setError(null);
        
        // Request initial state - use wsRef to avoid closure issues
        const message = {
          type: 'command',
          request_id: Date.now().toString(),
          command: 'get_state'
        };
        ws.send(JSON.stringify(message));
      };

      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data);
          
          switch (message.type) {
            case 'state_update':
              const stateData = message.data;
              
              // Use functional updates to avoid stale closure issues
              setWorldTime(prev => stateData.world_time !== prev ? stateData.world_time : prev);
              
              setSimulacraProfiles(prev => {
                const newProfiles = stateData.simulacra_profiles || {};
                return JSON.stringify(newProfiles) !== JSON.stringify(prev) ? newProfiles : prev;
              });
              
              setActiveSimulacraIds(prev => {
                const newIds = stateData.active_simulacra_ids || [];
                return JSON.stringify(newIds) !== JSON.stringify(prev) ? newIds : prev;
              });
              
              setWorldFeeds(prev => {
                const newFeeds = stateData.world_feeds || {};
                return JSON.stringify(newFeeds) !== JSON.stringify(prev) ? newFeeds : prev;
              });
              
              setNarrativeLog(prev => {
                const newLog = stateData.narrative_log || [];
                return JSON.stringify(newLog) !== JSON.stringify(prev) ? newLog : prev;
              });
              
              setWorldState(prev => {
                const newWorldState = {
                  current_world_state: stateData.current_world_state,
                  world_instance_uuid: stateData.world_instance_uuid,
                  world_template_details: stateData.world_template_details
                };
                return JSON.stringify(newWorldState) !== JSON.stringify(prev) ? newWorldState : prev;
              });
              
              // Extract events from simulation state
              extractEventsFromState(stateData);
              break;
              
            case 'event':
              // Convert event data to our EventData format
              const eventData: EventData = {
                timestamp: Date.now() / 1000,
                sim_time_s: message.data.timestamp || 0,
                agent_id: message.data.agent_id || 'System',
                event_type: message.data.event_type || 'system',
                data: message.data
              };
              setEvents(prev => [eventData, ...prev.slice(0, 99)]); // Keep last 100 events
              break;
              
            case 'command_response':
              if (message.request_id) {
                const callback = pendingCommands.current.get(message.request_id);
                if (callback) {
                  callback(message.response);
                  pendingCommands.current.delete(message.request_id);
                }
              }
              break;
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setConnected(false);
        
        // Attempt to reconnect after 3 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, 3000);
      };

      ws.onerror = (err) => {
        console.error('WebSocket error:', err);
        setError('WebSocket connection error');
      };

    } catch (err) {
      console.error('Failed to create WebSocket connection:', err);
      setError('Failed to connect to simulation');
    }
  }, [extractEventsFromState]);

  const sendCommand = useCallback((command: string, params: any): Promise<CommandResponse> => {
    return new Promise((resolve, reject) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        reject(new Error('WebSocket not connected'));
        return;
      }

      const requestId = Date.now().toString() + Math.random().toString(36).substr(2, 9);
      
      // Store the callback for this request
      pendingCommands.current.set(requestId, resolve);
      
      // Set timeout for command
      setTimeout(() => {
        if (pendingCommands.current.has(requestId)) {
          pendingCommands.current.delete(requestId);
          reject(new Error('Command timeout'));
        }
      }, 10000); // 10 second timeout

      const message = {
        type: 'command',
        request_id: requestId,
        command,
        ...params
      };

      wsRef.current.send(JSON.stringify(message));
    });
  }, []);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setConnected(false);
  }, []);

  // Command helpers
  const injectNarrative = useCallback(async (text: string) => {
    return sendCommand('inject_narrative', { text });
  }, [sendCommand]);

  const sendAgentEvent = useCallback(async (agent_id: string, description: string) => {
    return sendCommand('send_agent_event', { agent_id, description });
  }, [sendCommand]);

  const updateWorldInfo = useCallback(async (category: string, info: string) => {
    return sendCommand('update_world_info', { category, info });
  }, [sendCommand]);

  const teleportAgent = useCallback(async (agent_id: string, location: string) => {
    return sendCommand('teleport_agent', { agent_id, location });
  }, [sendCommand]);

  const interactiveChat = useCallback(async (agent_id: string, message: string) => {
    return sendCommand('interaction_event', { 
      agent_id, 
      content: message,
      event_type: 'telepathy'
    });
  }, [sendCommand]);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, []);

  
  return {
    state,
    events,
    connected,
    error,
    connect,
    disconnect,
    sendCommand,
    // Command helpers
    injectNarrative,
    sendAgentEvent,
    updateWorldInfo,
    teleportAgent,
    interactiveChat
  };
};