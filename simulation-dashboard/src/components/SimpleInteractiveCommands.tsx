import React, { useState } from 'react';
import { SimulationState } from '../types/simulation';
import { 
  MessageSquare, 
  Send, 
  Zap, 
  MapPin, 
  Cloud,
  Users,
  Terminal
} from 'lucide-react';

interface SimpleInteractiveCommandsProps {
  state: SimulationState;
  onInjectNarrative: (text: string) => Promise<any>;
  onSendAgentEvent: (agentId: string, description: string) => Promise<any>;
  onUpdateWorldInfo: (category: string, info: string) => Promise<any>;
  onTeleportAgent: (agentId: string, location: string) => Promise<any>;
  onInteractiveChat: (agentId: string, message: string) => Promise<any>;
  setInputFocus: (active: boolean) => void;
}

export const SimpleInteractiveCommands: React.FC<SimpleInteractiveCommandsProps> = ({ 
  state, 
  onInjectNarrative,
  onSendAgentEvent,
  onUpdateWorldInfo,
  onTeleportAgent,
  onInteractiveChat,
  setInputFocus
}) => {
  const [activeModal, setActiveModal] = useState<string | null>(null);
  const [narrativeText, setNarrativeText] = useState('');
  const [eventDescription, setEventDescription] = useState('');
  const [worldInfoText, setWorldInfoText] = useState('');
  const [worldInfoCategory, setWorldInfoCategory] = useState('weather');
  const [teleportLocation, setTeleportLocation] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [selectedAgent, setSelectedAgent] = useState('');

  const handleNarrativeSubmit = async () => {
    if (narrativeText.trim()) {
      try {
        await onInjectNarrative(narrativeText.trim());
        setNarrativeText('');
        setActiveModal(null);
      } catch (error) {
        console.error('Failed to inject narrative:', error);
      }
    }
  };

  const handleAgentEventSubmit = async () => {
    if (selectedAgent && eventDescription.trim()) {
      try {
        await onSendAgentEvent(selectedAgent, eventDescription.trim());
        setEventDescription('');
        setSelectedAgent('');
        setActiveModal(null);
      } catch (error) {
        console.error('Failed to send agent event:', error);
      }
    }
  };

  const handleWorldInfoSubmit = async () => {
    if (worldInfoText.trim()) {
      try {
        await onUpdateWorldInfo(worldInfoCategory, worldInfoText.trim());
        setWorldInfoText('');
        setActiveModal(null);
      } catch (error) {
        console.error('Failed to update world info:', error);
      }
    }
  };

  const handleTeleportSubmit = async () => {
    if (selectedAgent && teleportLocation.trim()) {
      try {
        await onTeleportAgent(selectedAgent, teleportLocation.trim());
        setTeleportLocation('');
        setSelectedAgent('');
        setActiveModal(null);
      } catch (error) {
        console.error('Failed to teleport agent:', error);
      }
    }
  };

  const handleChatSubmit = async () => {
    if (selectedAgent && chatInput.trim()) {
      try {
        await onInteractiveChat(selectedAgent, chatInput.trim());
        setChatInput('');
      } catch (error) {
        console.error('Failed to send chat message:', error);
      }
    }
  };

  const getAgentList = () => {
    if (!state?.simulacra_profiles) return [];
    return Object.entries(state.simulacra_profiles).map(([agentId, agentData]: [string, any]) => ({
      id: agentId,
      name: agentData.persona_details?.Name || agentId,
      status: agentData.status
    }));
  };

  return (
    <div className="bg-white rounded-lg shadow-lg border border-gray-200">
      <div className="p-6 border-b border-gray-200">
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          <Terminal className="w-5 h-5" />
          Interactive Commands
        </h2>
      </div>
      
      <div className="p-6">
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => setActiveModal('narrative')}
            className="flex items-center gap-2 px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <MessageSquare className="w-4 h-4" />
            <span className="text-sm">Inject Narrative</span>
          </button>
          
          <button
            onClick={() => setActiveModal('agent-event')}
            className="flex items-center gap-2 px-4 py-3 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
          >
            <Zap className="w-4 h-4" />
            <span className="text-sm">Send Agent Event</span>
          </button>
          
          <button
            onClick={() => setActiveModal('world-info')}
            className="flex items-center gap-2 px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <Cloud className="w-4 h-4" />
            <span className="text-sm">Update World Info</span>
          </button>
          
          <button
            onClick={() => setActiveModal('teleport')}
            className="flex items-center gap-2 px-4 py-3 bg-orange-600 text-white rounded-lg hover:bg-orange-700 transition-colors"
          >
            <MapPin className="w-4 h-4" />
            <span className="text-sm">Teleport Agent</span>
          </button>

          <button
            onClick={() => setActiveModal('chat')}
            className="flex items-center gap-2 px-4 py-3 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors"
          >
            <Users className="w-4 h-4" />
            <span className="text-sm">Interactive Chat</span>
          </button>
        </div>

        {/* Agent Status Preview */}
        {state && state.simulacra_profiles && (
          <div className="mt-6">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Active Agents</h3>
            <div className="space-y-1">
              {Object.entries(state.simulacra_profiles).slice(0, 3).map(([agentId, agentData]: [string, any]) => (
                <div key={agentId} className="text-xs p-2 bg-gray-50 rounded border-l-2 border-blue-300">
                  <div className="font-medium">{agentData.persona_details?.Name || agentId}</div>
                  <div className="text-gray-600">{agentData.status}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Modals */}
      {activeModal === 'narrative' && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-bold mb-4">Inject Narrative</h3>
            <textarea
              value={narrativeText}
              onChange={(e) => setNarrativeText(e.target.value)}
              className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-vertical"
              placeholder="Enter narrative text..."
              autoFocus
            />
            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setActiveModal(null)}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleNarrativeSubmit}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                Inject
              </button>
            </div>
          </div>
        </div>
      )}

      {activeModal === 'agent-event' && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-bold mb-4">Send Agent Event</h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Select Agent:</label>
              <select
                value={selectedAgent}
                onChange={(e) => setSelectedAgent(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg"
              >
                <option value="">Choose an agent...</option>
                {getAgentList().map(agent => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name} ({agent.status})
                  </option>
                ))}
              </select>
            </div>

            <textarea
              value={eventDescription}
              onChange={(e) => setEventDescription(e.target.value)}
              className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-vertical"
              placeholder="Describe the event happening to the agent..."
            />
            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setActiveModal(null)}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleAgentEventSubmit}
                className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
              >
                Send Event
              </button>
            </div>
          </div>
        </div>
      )}

      {activeModal === 'world-info' && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-bold mb-4">Update World Info</h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Category:</label>
              <select
                value={worldInfoCategory}
                onChange={(e) => setWorldInfoCategory(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg"
              >
                <option value="weather">Weather</option>
                <option value="news">News</option>
                <option value="local_news">Local News</option>
                <option value="pop_culture">Pop Culture</option>
              </select>
            </div>

            <textarea
              value={worldInfoText}
              onChange={(e) => setWorldInfoText(e.target.value)}
              className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-vertical"
              placeholder="Enter new world information..."
            />
            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setActiveModal(null)}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleWorldInfoSubmit}
                className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
              >
                Update
              </button>
            </div>
          </div>
        </div>
      )}

      {activeModal === 'teleport' && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-bold mb-4">Teleport Agent</h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Select Agent:</label>
              <select
                value={selectedAgent}
                onChange={(e) => setSelectedAgent(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg"
              >
                <option value="">Choose an agent...</option>
                {getAgentList().map(agent => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name} ({agent.status})
                  </option>
                ))}
              </select>
            </div>

            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Location:</label>
              <input
                type="text"
                value={teleportLocation}
                onChange={(e) => setTeleportLocation(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg"
                placeholder="e.g., Kitchen_Home_01, Park_Central..."
              />
            </div>

            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setActiveModal(null)}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Cancel
              </button>
              <button
                onClick={handleTeleportSubmit}
                className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700"
              >
                Teleport
              </button>
            </div>
          </div>
        </div>
      )}

      {activeModal === 'chat' && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-md mx-4">
            <h3 className="text-lg font-bold mb-4">Interactive Chat</h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">Select Agent:</label>
              <select
                value={selectedAgent}
                onChange={(e) => setSelectedAgent(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-lg"
              >
                <option value="">Choose an agent...</option>
                {getAgentList().map(agent => (
                  <option key={agent.id} value={agent.id}>
                    {agent.name} ({agent.status})
                  </option>
                ))}
              </select>
            </div>

            <div className="mb-4">
              <input
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && handleChatSubmit()}
                className="w-full p-3 border border-gray-300 rounded-lg"
                placeholder="Type your message..."
              />
            </div>

            <div className="flex gap-2 mt-4">
              <button
                onClick={() => setActiveModal(null)}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
              >
                Close
              </button>
              <button
                onClick={handleChatSubmit}
                className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
              >
                Send
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};