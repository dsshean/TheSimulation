import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { EventData } from '../types/simulation';
import { getEventTypeColor, getEventTypeIcon, formatTime } from '../utils/formatters';
import { ScrollArea } from './ScrollArea';

interface EventLogProps {
  events: EventData[];
  title?: string;
}

export const EventLog: React.FC<EventLogProps> = ({ events, title = "Event Log" }) => {
  const renderEventContent = (event: EventData) => {
    const { event_type, data } = event;
    
    switch (event_type) {
      case 'monologue':
        return (
          <div className="space-y-2">
            <div className="font-medium text-blue-900">💭 Internal Thoughts</div>
            <div className="text-sm text-blue-800 bg-blue-50 p-3 rounded">
              {data.monologue}
            </div>
          </div>
        );
        
      case 'intent':
        return (
          <div className="space-y-2">
            <div className="font-medium text-yellow-900">🎯 Action Intent</div>
            <div className="text-sm text-yellow-800 bg-yellow-50 p-3 rounded space-y-1">
              <div><strong>Action:</strong> {data.action_type}</div>
              {data.target_id && <div><strong>Target:</strong> {data.target_id}</div>}
              {data.details && <div><strong>Details:</strong> {data.details}</div>}
            </div>
          </div>
        );
        
      case 'resolution':
        const isValid = data.valid_action;
        const colorClass = isValid ? 'text-green-900' : 'text-red-900';
        const bgColorClass = isValid ? 'bg-green-50' : 'bg-red-50';
        
        return (
          <div className="space-y-2">
            <div className={`font-medium ${colorClass}`}>
              {isValid ? '✅' : '❌'} World Engine Resolution
            </div>
            <div className={`text-sm ${colorClass} ${bgColorClass} p-3 rounded space-y-2`}>
              <div><strong>Duration:</strong> {data.duration}s</div>
              <div><strong>Outcome:</strong> {data.outcome_description}</div>
              
              {data.results?.discovered_objects?.length > 0 && (
                <div>
                  <strong>Objects Found:</strong> {data.results.discovered_objects.length}
                  {data.results.discovered_objects.slice(0, 3).map((obj: any, idx: number) => (
                    <div key={idx} className="ml-4 text-xs">
                      • {obj.name}: {obj.description}
                    </div>
                  ))}
                </div>
              )}
              
              {data.results?.discovered_connections?.length > 0 && (
                <div>
                  <strong>Connections Found:</strong> {data.results.discovered_connections.length}
                  {data.results.discovered_connections.slice(0, 3).map((conn: any, idx: number) => (
                    <div key={idx} className="ml-4 text-xs">
                      • To {conn.to_location_id_hint}: {conn.description}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
        
      case 'narrative':
        return (
          <div className="space-y-2">
            <div className="font-medium text-purple-900">📖 Narrative</div>
            <div className="text-sm text-purple-800 bg-purple-50 p-3 rounded">
              {data.narrative_text}
            </div>
          </div>
        );
        
      case 'world_engine':
        return (
          <div className="space-y-2">
            <div className="font-medium text-green-900">🌍 World Engine</div>
            <div className="text-sm text-green-800 bg-green-50 p-3 rounded space-y-2">
              <div>{data.content || data.outcome_description || 'World Engine processing...'}</div>
              {data.details && (
                <div className="text-xs text-green-700 mt-2 p-2 bg-green-100 rounded">
                  <pre className="whitespace-pre-wrap">{data.details}</pre>
                </div>
              )}
            </div>
          </div>
        );
        
      default:
        return (
          <div className="space-y-2">
            <div className="font-medium text-gray-900">Unknown Event</div>
            <div className="text-sm text-gray-800 bg-gray-50 p-3 rounded">
              {JSON.stringify(data, null, 2)}
            </div>
          </div>
        );
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200 h-full flex flex-col">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800">{title}</h2>
      </div>
      
      <ScrollArea className="flex-1 p-6">
        <AnimatePresence>
          {events.length === 0 ? (
            <div className="text-center py-8 text-gray-500">
              <div className="text-4xl mb-4">📄</div>
              <p>No events yet</p>
            </div>
          ) : (
            <div className="space-y-4">
              {events.map((event, index) => (
                <motion.div
                  key={`${event.timestamp}-${index}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ delay: index * 0.05 }}
                  className={`border-l-4 ${getEventTypeColor(event.event_type)} bg-gray-50 p-4 rounded-r-lg`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <span className="text-lg">
                        {getEventTypeIcon(event.event_type)}
                      </span>
                      <span className="font-medium text-gray-900">
                        {event.agent_id}
                      </span>
                    </div>
                    <span className="text-xs text-gray-500">
                      T{formatTime(event.sim_time_s)}
                    </span>
                  </div>
                  
                  {renderEventContent(event)}
                </motion.div>
              ))}
            </div>
          )}
        </AnimatePresence>
      </ScrollArea>
    </div>
  );
};