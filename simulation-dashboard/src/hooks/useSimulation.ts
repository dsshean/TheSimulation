import { useWebSocket } from './useWebSocket';

export const useSimulation = () => {
  const {
    state,
    events,
    connected,
    error: wsError,
    connect,
    injectNarrative,
    sendAgentEvent,
    updateWorldInfo,
    teleportAgent,
    interactiveChat
  } = useWebSocket();

  return {
    state,
    events,
    loading: !connected && !wsError,
    error: wsError,
    connected,
    refresh: connect,
    // Command functions
    injectNarrative,
    sendAgentEvent,
    updateWorldInfo,
    teleportAgent,
    interactiveChat
  };
};