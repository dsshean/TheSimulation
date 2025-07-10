import React from 'react';
import { motion } from 'framer-motion';
import { SimulationState } from '../types/simulation';
import { getStatusColor, getStatusEmoji, formatTime } from '../utils/formatters';
import { User, Clock, Target, MapPin, Package, Users, ArrowRight } from 'lucide-react';

interface AgentLocationStatusProps {
  state: SimulationState;
}

export const AgentLocationStatus: React.FC<AgentLocationStatusProps> = React.memo(({ state }) => {
  const simulacra = state.simulacra_profiles || {};
  const activeIds = state.active_simulacra_ids || [];
  const currentTime = state.world_time;
  const locationDetails = state.current_world_state?.location_details || {};

  if (activeIds.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-white rounded-lg shadow-lg border border-gray-200"
      >
        <div className="p-4 border-b border-gray-200">
          <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
            <User className="w-5 h-5" />
            Agent Status & Location
          </h2>
        </div>
        <div className="p-6">
          <div className="text-center py-8 text-gray-500">
            <User className="w-12 h-12 mx-auto mb-4 text-gray-300" />
            <p>No active agents</p>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="bg-white rounded-lg shadow-lg border border-gray-200"
    >
      <div className="p-4 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <User className="w-5 h-5" />
          Agent Status & Location
        </h2>
      </div>
      
      <div className="p-4 overflow-y-auto max-h-80">
        <div className="space-y-4">
          {activeIds.map((id) => {
            const agent = simulacra[id];
            if (!agent) return null;
            
            const name = agent.persona_details?.Name || id;
            const status = agent.status || 'unknown';
            const action = agent.current_action_description || 'Idle';
            const endTime = agent.current_action_end_time || 0;
            const locationId = agent.current_location;
            const location = locationId ? locationDetails[locationId] : null;
            
            let durationText = '';
            if (status === 'busy' && endTime > currentTime) {
              const remaining = endTime - currentTime;
              durationText = `${formatTime(remaining)} left`;
            } else if (status === 'busy') {
              durationText = 'Completing...';
            }
            
            return (
              <motion.div
                key={id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className="border rounded-lg bg-gray-50 overflow-hidden"
              >
                {/* Agent Status Header */}
                <div className="p-4 bg-white border-b">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      <div className={`w-3 h-3 rounded-full ${getStatusColor(status)}`} />
                      <span className="font-medium text-gray-900">{name}</span>
                    </div>
                    <div className="text-sm text-gray-600">
                      {getStatusEmoji(status)} {status}
                    </div>
                  </div>
                  
                  <div className="flex items-start gap-2 mb-2">
                    <Target className="w-4 h-4 text-gray-500 mt-0.5" />
                    <p className="text-sm text-gray-700 flex-1">{action}</p>
                  </div>
                  
                  {durationText && (
                    <div className="flex items-center gap-2 text-xs text-gray-500">
                      <Clock className="w-3 h-3" />
                      <span>{durationText}</span>
                    </div>
                  )}
                </div>

                {/* Location Info */}
                <div className="p-4">
                  {location ? (
                    <div className="space-y-3">
                      <div className="flex items-center gap-2">
                        <MapPin className="w-4 h-4 text-blue-600" />
                        <div>
                          <div className="font-medium text-blue-900">{location.name || locationId}</div>
                          {location.description && (
                            <div className="text-sm text-gray-600">{location.description}</div>
                          )}
                        </div>
                      </div>

                      {/* Quick Stats */}
                      <div className="flex items-center gap-4 text-xs text-gray-500">
                        {location.ephemeral_objects && location.ephemeral_objects.length > 0 && (
                          <div className="flex items-center gap-1">
                            <Package className="w-3 h-3" />
                            <span>{location.ephemeral_objects.length} objects</span>
                          </div>
                        )}
                        {location.ephemeral_npcs && location.ephemeral_npcs.length > 0 && (
                          <div className="flex items-center gap-1">
                            <Users className="w-3 h-3" />
                            <span>{location.ephemeral_npcs.length} NPCs</span>
                          </div>
                        )}
                        {location.connected_locations && location.connected_locations.length > 0 && (
                          <div className="flex items-center gap-1">
                            <ArrowRight className="w-3 h-3" />
                            <span>{location.connected_locations.length} exits</span>
                          </div>
                        )}
                      </div>

                      {/* Notable Objects */}
                      {location.ephemeral_objects && location.ephemeral_objects.length > 0 && (
                        <div className="space-y-1">
                          <div className="text-xs font-medium text-gray-700">Objects:</div>
                          <div className="flex flex-wrap gap-1">
                            {location.ephemeral_objects.slice(0, 10).map((obj, index) => (
                              <span key={index} className="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">
                                {obj.name}
                              </span>
                            ))}
                            {location.ephemeral_objects.length > 10 && (
                              <span className="text-xs text-gray-500">
                                +{location.ephemeral_objects.length - 10} more
                              </span>
                            )}
                          </div>
                        </div>
                      )}

                      {/* Exits */}
                      {location.connected_locations && location.connected_locations.length > 0 && (
                        <div className="space-y-1">
                          <div className="text-xs font-medium text-gray-700">Exits:</div>
                          <div className="flex flex-wrap gap-1">
                            {location.connected_locations.slice(0, 10).map((conn, index) => (
                              <span key={index} className="text-xs bg-purple-100 text-purple-800 px-2 py-1 rounded">
                                {conn.to_location_id_hint}
                              </span>
                            ))}
                            {location.connected_locations.length > 10 && (
                              <span className="text-xs text-gray-500">
                                +{location.connected_locations.length - 10} more
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                    </div>
                  ) : locationId ? (
                    <div className="flex items-center gap-2 text-gray-500">
                      <MapPin className="w-4 h-4" />
                      <span className="text-sm">{locationId}</span>
                    </div>
                  ) : (
                    <div className="flex items-center gap-2 text-gray-500">
                      <MapPin className="w-4 h-4" />
                      <span className="text-sm">Location unknown</span>
                    </div>
                  )}
                </div>
              </motion.div>
            );
          })}
        </div>
      </div>
    </motion.div>
  );
}, (prevProps, nextProps) => {
  // Only re-render if relevant data has changed
  return (
    JSON.stringify(prevProps.state.simulacra_profiles) === JSON.stringify(nextProps.state.simulacra_profiles) &&
    JSON.stringify(prevProps.state.active_simulacra_ids) === JSON.stringify(nextProps.state.active_simulacra_ids) &&
    JSON.stringify(prevProps.state.current_world_state?.location_details) === JSON.stringify(nextProps.state.current_world_state?.location_details) &&
    prevProps.state.world_time === nextProps.state.world_time
  );
});