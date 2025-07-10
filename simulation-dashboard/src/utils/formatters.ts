export const formatTime = (seconds: number): string => {
  if (seconds < 60) {
    return `${seconds.toFixed(1)}s`;
  } else if (seconds < 3600) {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}m ${remainingSeconds.toFixed(1)}s`;
  } else {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const remainingSeconds = seconds % 60;
    return `${hours}h ${minutes}m ${remainingSeconds.toFixed(0)}s`;
  }
};

export const formatEventTime = (timestamp: number): string => {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString();
};

export const getStatusColor = (status: string): string => {
  switch (status) {
    case 'idle':
      return 'bg-green-500';
    case 'busy':
      return 'bg-orange-500';
    case 'thinking':
      return 'bg-yellow-500';
    default:
      return 'bg-gray-500';
  }
};

export const getStatusEmoji = (status: string): string => {
  switch (status) {
    case 'idle':
      return '🟢';
    case 'busy':
      return '🟡';
    case 'thinking':
      return '🔴';
    default:
      return '⚪';
  }
};

export const getEventTypeColor = (eventType: string): string => {
  switch (eventType) {
    case 'monologue':
      return 'border-blue-500';
    case 'intent':
      return 'border-yellow-500';
    case 'resolution':
      return 'border-green-500';
    case 'narrative':
      return 'border-purple-500';
    default:
      return 'border-gray-500';
  }
};

export const getEventTypeIcon = (eventType: string): string => {
  switch (eventType) {
    case 'monologue':
      return '💭';
    case 'intent':
      return '🎯';
    case 'resolution':
      return '✅';
    case 'narrative':
      return '📖';
    default:
      return '📄';
  }
};