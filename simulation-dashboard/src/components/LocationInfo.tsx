import React from 'react';
import { motion } from 'framer-motion';
import { SimulationState } from '../types/simulation';
import { MapPin, Package, Users, ArrowRight } from 'lucide-react';

interface LocationInfoProps {
  state: SimulationState;
}

export const LocationInfo: React.FC<LocationInfoProps> = ({ state }) => {
  const simulacra = state.simulacra_profiles || {};
  const activeIds = state.active_simulacra_ids || [];
  
  // Get primary agent's location
  const primaryAgent = activeIds.length > 0 ? simulacra[activeIds[0]] : null;
  const locationId = primaryAgent?.current_location;
  const locationDetails = state.current_world_state?.location_details || {};
  const location = locationId ? locationDetails[locationId] : null;

  if (!location) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white rounded-lg shadow-lg border border-gray-200 p-6"
      >
        <div className="text-center py-8 text-gray-500">
          <MapPin className="w-12 h-12 mx-auto mb-4 text-gray-300" />
          <p>No location data available</p>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white rounded-lg shadow-lg border border-gray-200"
    >
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <MapPin className="w-5 h-5" />
          Current Location
        </h2>
        <p className="text-lg font-semibold text-blue-600 mt-1">
          {location.name}
        </p>
        {location.description && (
          <p className="text-sm text-gray-600 mt-2">{location.description}</p>
        )}
      </div>
      
      <div className="p-6 space-y-6">
        {/* Objects */}
        {location.ephemeral_objects && location.ephemeral_objects.length > 0 && (
          <div>
            <h3 className="font-medium text-gray-700 mb-3 flex items-center gap-2">
              <Package className="w-4 h-4" />
              Objects ({location.ephemeral_objects.length})
            </h3>
            <div className="space-y-2">
              {location.ephemeral_objects.slice(0, 5).map((obj, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-3 bg-blue-50 rounded-lg"
                >
                  <div className="font-medium text-blue-900">{obj.name}</div>
                  <div className="text-sm text-blue-700">{obj.description}</div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
        
        {/* NPCs */}
        {location.ephemeral_npcs && location.ephemeral_npcs.length > 0 && (
          <div>
            <h3 className="font-medium text-gray-700 mb-3 flex items-center gap-2">
              <Users className="w-4 h-4" />
              NPCs ({location.ephemeral_npcs.length})
            </h3>
            <div className="space-y-2">
              {location.ephemeral_npcs.slice(0, 3).map((npc, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-3 bg-green-50 rounded-lg"
                >
                  <div className="font-medium text-green-900">👤 {npc.name}</div>
                  <div className="text-sm text-green-700">{npc.description}</div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
        
        {/* Connections */}
        {location.connected_locations && location.connected_locations.length > 0 && (
          <div>
            <h3 className="font-medium text-gray-700 mb-3 flex items-center gap-2">
              <ArrowRight className="w-4 h-4" />
              Connections ({location.connected_locations.length})
            </h3>
            <div className="space-y-2">
              {location.connected_locations.slice(0, 5).map((conn, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="p-3 bg-purple-50 rounded-lg flex items-center gap-2"
                >
                  <ArrowRight className="w-4 h-4 text-purple-600" />
                  <div className="flex-1">
                    <div className="font-medium text-purple-900">
                      {conn.to_location_id_hint}
                    </div>
                    <div className="text-sm text-purple-700">{conn.description}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  );
};