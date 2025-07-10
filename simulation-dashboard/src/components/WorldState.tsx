import React from 'react';
import { motion } from 'framer-motion';
import { SimulationState } from '../types/simulation';
import { formatTime } from '../utils/formatters';
import { Clock, Globe, Users, Activity } from 'lucide-react';

interface WorldStateProps {
  state: SimulationState;
}

export const WorldState: React.FC<WorldStateProps> = React.memo(({ state }) => {
  const weather = state.world_feeds?.weather?.condition || 'Unknown';
  const activeCount = state.active_simulacra_ids?.length || 0;
  
  return (
    <motion.div
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-lg p-6 border border-gray-200"
    >
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <Globe className="w-5 h-5" />
          World State
        </h2>
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-green-500" />
          <span className="text-sm text-green-600 font-medium">Live</span>
        </div>
      </div>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-blue-50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Clock className="w-4 h-4 text-blue-600" />
            <span className="text-sm font-medium text-blue-800">Time</span>
          </div>
          <p className="text-lg font-bold text-blue-900">
            {formatTime(state.world_time)}
          </p>
        </div>
        
        <div className="bg-green-50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-medium text-green-800">Weather</span>
          </div>
          <p className="text-lg font-bold text-green-900">
            🌡️ {weather}
          </p>
        </div>
        
        <div className="bg-purple-50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <Users className="w-4 h-4 text-purple-600" />
            <span className="text-sm font-medium text-purple-800">Agents</span>
          </div>
          <p className="text-lg font-bold text-purple-900">
            {activeCount}
          </p>
        </div>
        
        <div className="bg-orange-50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-sm font-medium text-orange-800">UUID</span>
          </div>
          <p className="text-sm font-mono text-orange-900">
            {state.world_instance_uuid?.slice(0, 8) || 'N/A'}...
          </p>
        </div>
      </div>
      
      {state.world_template_details?.description && (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg">
          <h3 className="font-medium text-gray-700 mb-2">World Description</h3>
          <p className="text-sm text-gray-600">
            {state.world_template_details.description}
          </p>
        </div>
      )}
    </motion.div>
  );
}, (prevProps, nextProps) => {
  // Only re-render if the relevant parts of state have changed
  return (
    prevProps.state.world_time === nextProps.state.world_time &&
    prevProps.state.world_feeds?.weather?.condition === nextProps.state.world_feeds?.weather?.condition &&
    prevProps.state.active_simulacra_ids?.length === nextProps.state.active_simulacra_ids?.length &&
    prevProps.state.world_instance_uuid === nextProps.state.world_instance_uuid &&
    prevProps.state.world_template_details?.description === nextProps.state.world_template_details?.description
  );
});