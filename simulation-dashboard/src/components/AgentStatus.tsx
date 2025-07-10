import React from 'react';
import { motion } from 'framer-motion';
import { SimulationState } from '../types/simulation';
import { getStatusColor, getStatusEmoji, formatTime } from '../utils/formatters';
import { User, Clock, Target } from 'lucide-react';

interface AgentStatusProps {
  state: SimulationState;
}

export const AgentStatus: React.FC<AgentStatusProps> = React.memo(({ state }) => {
  const simulacra = state.simulacra_profiles || {};
  const activeIds = state.active_simulacra_ids || [];
  const currentTime = state.world_time;

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="bg-white rounded-lg shadow-lg border border-gray-200 h-full"
    >
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <User className="w-5 h-5" />
          Agent Status
        </h2>
      </div>
      
      <div className="p-6 overflow-y-auto max-h-96">
        {activeIds.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <User className="w-12 h-12 mx-auto mb-4 text-gray-300" />
            <p>No active agents</p>
          </div>
        ) : (
          <div className="space-y-4">
            {activeIds.map((id) => {
              const agent = simulacra[id];
              if (!agent) return null;
              
              const name = agent.persona_details?.Name || id;
              const status = agent.status || 'unknown';
              const action = agent.current_action_description || 'Idle';
              const endTime = agent.current_action_end_time || 0;
              
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
                  className="p-4 border rounded-lg bg-gray-50 hover:bg-gray-100 transition-colors"
                >
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
                </motion.div>
              );
            })}
          </div>
        )}
      </div>
    </motion.div>
  );
}, (prevProps, nextProps) => {
  // Only re-render if simulacra data has changed
  return (
    JSON.stringify(prevProps.state.simulacra_profiles) === JSON.stringify(nextProps.state.simulacra_profiles) &&
    JSON.stringify(prevProps.state.active_simulacra_ids) === JSON.stringify(nextProps.state.active_simulacra_ids) &&
    prevProps.state.world_time === nextProps.state.world_time
  );
});