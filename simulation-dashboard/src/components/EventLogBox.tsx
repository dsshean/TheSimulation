import React, { useState } from 'react';
import { EventData } from '../types/simulation';
import { ChevronDown, ChevronRight } from 'lucide-react';

interface EventLogBoxProps {
  title: string;
  events: EventData[];
  icon: React.ReactNode;
  bgColor?: string;
  textColor?: string;
}

export const EventLogBox: React.FC<EventLogBoxProps> = React.memo(({ 
  title, 
  events, 
  icon, 
  bgColor = 'bg-white',
  textColor = 'text-gray-800'
}) => {
  // Get the most recent event
  const latestEvent = events.length > 0 ? events[0] : null;
  const [expandedEvents, setExpandedEvents] = useState<Set<number>>(new Set());

  const formatEventContent = (event: EventData) => {
    if (event.event_type === 'monologue') {
      return event.data.content || event.data.monologue || 'Agent thinking...';
    } else if (event.event_type === 'observation') {
      return event.data.content || event.data.observation || 'Agent observing...';
    } else if (event.event_type === 'intent') {
      return `Action: ${event.data.action_type || 'unknown'} - ${event.data.details || ''}`;
    } else if (event.event_type === 'resolution') {
      return event.data.outcome_description || 'World engine processing...';
    } else if (event.event_type === 'narrative') {
      return event.data.narrative_text || event.data.content || 'Narrative update...';
    }
    return JSON.stringify(event.data);
  };

  const formatTime = (timestamp: number) => {
    return new Date(timestamp * 1000).toLocaleTimeString();
  };

  const toggleEventExpansion = (index: number) => {
    const newExpanded = new Set(expandedEvents);
    if (newExpanded.has(index)) {
      newExpanded.delete(index);
    } else {
      newExpanded.add(index);
    }
    setExpandedEvents(newExpanded);
  };

  const getPreviewText = (content: string, maxLength: number = 80) => {
    if (content.length <= maxLength) return content;
    return content.substring(0, maxLength) + '...';
  };

  return (
    <div className={`${bgColor} rounded-lg shadow-lg border border-gray-200 h-96`}>
      <div className={`p-4 border-b border-gray-200 ${textColor}`}>
        <h3 className="text-lg font-bold flex items-center gap-2">
          {icon}
          {title}
        </h3>
        <p className="text-sm opacity-60">Latest: {events.length} total events</p>
      </div>
      
      <div className="p-4 h-80 overflow-y-auto">
        {latestEvent ? (
          <div className="space-y-3">
            <div className={`p-3 rounded-lg border-l-4 ${
              latestEvent.event_type === 'monologue' ? 'border-blue-500 bg-blue-50' :
              latestEvent.event_type === 'observation' ? 'border-cyan-500 bg-cyan-50' :
              latestEvent.event_type === 'intent' ? 'border-purple-500 bg-purple-50' :
              latestEvent.event_type === 'resolution' ? 'border-green-500 bg-green-50' :
              latestEvent.event_type === 'narrative' ? 'border-orange-500 bg-orange-50' :
              'border-gray-500 bg-gray-50'
            }`}>
              <div className="flex justify-between items-start mb-2">
                <span className="text-xs font-medium text-gray-600">
                  {latestEvent.agent_id}
                </span>
                <span className="text-xs text-gray-500">
                  {formatTime(latestEvent.timestamp)}
                </span>
              </div>
              <p className="text-sm text-gray-800 leading-relaxed">
                {formatEventContent(latestEvent)}
              </p>
            </div>
            
            {/* Show collapsible recent events */}
            {events.slice(1, 8).map((event, index) => {
              const eventIndex = index + 1; // Since we're starting from slice(1)
              const isExpanded = expandedEvents.has(eventIndex);
              const content = formatEventContent(event);
              
              return (
                <div key={eventIndex} className="bg-gray-50 rounded border-l-2 border-gray-300">
                  <div 
                    className="p-2 cursor-pointer hover:bg-gray-100 transition-colors"
                    onClick={() => toggleEventExpansion(eventIndex)}
                  >
                    <div className="flex justify-between items-center mb-1">
                      <div className="flex items-center gap-2">
                        {isExpanded ? 
                          <ChevronDown className="w-3 h-3 text-gray-500" /> : 
                          <ChevronRight className="w-3 h-3 text-gray-500" />
                        }
                        <span className="font-medium text-xs text-gray-600">{event.agent_id}</span>
                      </div>
                      <span className="text-xs text-gray-500">{formatTime(event.timestamp)}</span>
                    </div>
                    <p className="text-xs text-gray-600 ml-5">
                      {isExpanded ? content : getPreviewText(content)}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <div className="text-4xl mb-2">📭</div>
              <p>No {title.toLowerCase()} events yet</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}, (prevProps, nextProps) => {
  // Only re-render if events have changed
  return (
    prevProps.events.length === nextProps.events.length &&
    JSON.stringify(prevProps.events.slice(0, 5)) === JSON.stringify(nextProps.events.slice(0, 5))
  );
});