import React, { useState, useEffect } from 'react';
import { EventData } from '../types/simulation';
import { 
  MessageSquare, 
  Brain, 
  Activity, 
  Eye,
  Clock,
  Filter
} from 'lucide-react';

interface LiveEventFeedProps {
  events: EventData[];
  maxEvents?: number;
}

export const LiveEventFeed: React.FC<LiveEventFeedProps> = ({ 
  events, 
  maxEvents = 10 
}) => {
  const [filter, setFilter] = useState<string>('all');
  const [animate, setAnimate] = useState<string | null>(null);
  const [lastEventTimestamp, setLastEventTimestamp] = useState<number | null>(null);

  // Animate new events only when timestamp changes
  useEffect(() => {
    if (events.length > 0) {
      const latestEvent = events[0];
      const currentTimestamp = latestEvent.timestamp;
      
      // Only animate if this is a truly new event
      if (lastEventTimestamp !== currentTimestamp) {
        setAnimate(`${latestEvent.agent_id}-${currentTimestamp}`);
        setLastEventTimestamp(currentTimestamp);
        setTimeout(() => setAnimate(null), 1000);
      }
    }
  }, [events, lastEventTimestamp]);

  const getEventIcon = (eventType: string) => {
    switch (eventType) {
      case 'narrative': return <MessageSquare className="w-4 h-4 text-blue-500" />;
      case 'world_engine': return <Activity className="w-4 h-4 text-green-500" />;
      case 'monologue': return <Brain className="w-4 h-4 text-purple-500" />;
      case 'intent': return <Eye className="w-4 h-4 text-orange-500" />;
      case 'observation': return <Eye className="w-4 h-4 text-gray-500" />;
      default: return <Clock className="w-4 h-4 text-gray-400" />;
    }
  };

  const getEventColor = (eventType: string) => {
    switch (eventType) {
      case 'narrative': return 'border-l-blue-400 bg-blue-50';
      case 'world_engine': return 'border-l-green-400 bg-green-50';
      case 'monologue': return 'border-l-purple-400 bg-purple-50';
      case 'intent': return 'border-l-orange-400 bg-orange-50';
      case 'observation': return 'border-l-gray-400 bg-gray-50';
      default: return 'border-l-gray-300 bg-gray-50';
    }
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  const filteredEvents = filter === 'all' 
    ? events.slice(0, maxEvents)
    : events.filter(event => event.event_type === filter).slice(0, maxEvents);

  const eventTypes = ['all', 'narrative', 'world_engine', 'monologue', 'intent', 'observation'];

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200">
      <div className="p-4 border-b border-gray-200">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-gray-800 flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Live Event Feed
          </h3>
          
          <div className="flex items-center gap-2">
            <Filter className="w-4 h-4 text-gray-500" />
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="text-sm border border-gray-300 rounded px-2 py-1"
            >
              {eventTypes.map(type => (
                <option key={type} value={type}>
                  {type.charAt(0).toUpperCase() + type.slice(1).replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      <div className="max-h-96 overflow-y-auto">
        {filteredEvents.length === 0 ? (
          <div className="p-4 text-center text-gray-500">
            <Activity className="w-8 h-8 mx-auto mb-2 text-gray-300" />
            <p>No events yet...</p>
          </div>
        ) : (
          <div className="space-y-1 p-2">
            {filteredEvents.map((event, index) => {
              const eventKey = `${event.agent_id}-${event.timestamp}`;
              const isAnimated = animate === eventKey;
              
              return (
                <div
                  key={`${event.timestamp}-${index}`}
                  className={`
                    border-l-4 p-3 rounded-r-lg transition-all duration-300
                    ${getEventColor(event.event_type)}
                    ${isAnimated ? 'animate-pulse shadow-md scale-105' : 'hover:shadow-sm'}
                  `}
                >
                  <div className="flex items-start gap-2">
                    {getEventIcon(event.event_type)}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <span className="font-medium text-sm text-gray-800 truncate">
                          {event.agent_id}
                        </span>
                        <span className="text-xs text-gray-500 flex-shrink-0">
                          T{event.timestamp?.toFixed(1) || '0.0'}s
                        </span>
                      </div>
                      
                      <p className="text-xs text-gray-600 mt-1 leading-relaxed">
                        {event.data?.content || 'No content'}
                      </p>
                      
                      {event.data?.details && (
                        <p className="text-xs text-gray-500 mt-1 italic">
                          {event.data.details}
                        </p>
                      )}
                      
                      {event.data?.status && (
                        <span className={`inline-block px-2 py-1 text-xs rounded-full mt-1 ${
                          event.data.status === 'busy' ? 'bg-orange-100 text-orange-700' :
                          event.data.status === 'idle' ? 'bg-green-100 text-green-700' :
                          event.data.status === 'thinking' ? 'bg-blue-100 text-blue-700' :
                          'bg-gray-100 text-gray-700'
                        }`}>
                          {event.data.status}
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      <div className="p-2 border-t border-gray-200 bg-gray-50">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <span>Showing last {Math.min(filteredEvents.length, maxEvents)} events</span>
          <span>Live updates every 3s</span>
        </div>
      </div>
    </div>
  );
};