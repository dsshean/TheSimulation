import React, { useState, useCallback, useMemo } from 'react';
import { motion } from 'framer-motion';
import { useSimulation } from '../context/SimulationContext';
import { 
  MessageSquare, 
  Send, 
  Zap, 
  MapPin, 
  Cloud,
  History,
  Users,
  Terminal
} from 'lucide-react';

interface CommandHistory {
  id: string;
  timestamp: number;
  command: string;
  status: 'success' | 'error' | 'pending';
  response?: string;
}

export const IsolatedInteractiveCommands: React.FC = React.memo(() => {
  console.log('IsolatedInteractiveCommands rendered at:', new Date().toISOString());
  
  const { 
    state, 
    injectNarrative, 
    sendAgentEvent, 
    updateWorldInfo, 
    teleportAgent, 
    interactiveChat
  } = useSimulation();

  const [activeModal, setActiveModal] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string>('');
  const [commandHistory, setCommandHistory] = useState<CommandHistory[]>([]);
  
  // Form states - these are completely isolated
  const [narrativeText, setNarrativeText] = useState('');
  const [eventDescription, setEventDescription] = useState('');
  const [worldInfoCategory, setWorldInfoCategory] = useState('weather');
  const [worldInfoText, setWorldInfoText] = useState('');
  const [teleportLocation, setTeleportLocation] = useState('');
  const [teleportDescription, setTeleportDescription] = useState('');
  const [chatInput, setChatInput] = useState('');
  const [chatMessages, setChatMessages] = useState<Array<{type: 'user' | 'agent' | 'system'; content: string; timestamp: number}>>([]);

  

  const addToHistory = useCallback((command: string, status: 'success' | 'error' | 'pending' = 'pending', response?: string) => {
    const newEntry: CommandHistory = {
      id: Date.now().toString(),
      timestamp: Date.now(),
      command,
      status,
      response
    };
    setCommandHistory(prev => [newEntry, ...prev.slice(0, 49)]);
    return newEntry.id;
  }, []);

  const updateHistoryStatus = useCallback((id: string, status: 'success' | 'error', response?: string) => {
    setCommandHistory(prev => prev.map(entry => 
      entry.id === id ? { ...entry, status, response } : entry
    ));
  }, []);

  const executeCommand = useCallback(async (command: string, callback: () => Promise<any>) => {
    const historyId = addToHistory(command);
    
    try {
      const result = await callback();
      updateHistoryStatus(historyId, 'success', result.message || 'Command executed successfully');
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      updateHistoryStatus(historyId, 'error', errorMessage);
      return { success: false, message: errorMessage };
    }
  }, [addToHistory, updateHistoryStatus]);

  const handleNarrativeInjection = useCallback(async () => {
    if (!narrativeText.trim()) return;
    
    await executeCommand(
      `inject_narrative: ${narrativeText}`,
      () => injectNarrative(narrativeText)
    );
    setNarrativeText('');
    setActiveModal(null);
  }, [narrativeText, injectNarrative, executeCommand]);

  const handleAgentEvent = useCallback(async () => {
    if (!selectedAgent || !eventDescription.trim()) return;
    
    await executeCommand(
      `send_agent_event: ${selectedAgent} - ${eventDescription}`,
      () => sendAgentEvent(selectedAgent, eventDescription)
    );
    setEventDescription('');
    setSelectedAgent('');
    setActiveModal(null);
  }, [selectedAgent, eventDescription, sendAgentEvent, executeCommand]);

  const handleWorldInfoUpdate = useCallback(async () => {
    if (!worldInfoText.trim()) return;
    
    await executeCommand(
      `update_world_info: ${worldInfoCategory} - ${worldInfoText}`,
      () => updateWorldInfo(worldInfoCategory, worldInfoText)
    );
    setWorldInfoText('');
    setActiveModal(null);
  }, [worldInfoCategory, worldInfoText, updateWorldInfo, executeCommand]);

  const handleTeleport = useCallback(async () => {
    if (!selectedAgent || !teleportLocation.trim()) return;
    
    await executeCommand(
      `teleport_agent: ${selectedAgent} to ${teleportLocation}`,
      () => teleportAgent(selectedAgent, teleportLocation)
    );
    setTeleportLocation('');
    setTeleportDescription('');
    setSelectedAgent('');
    setActiveModal(null);
  }, [selectedAgent, teleportLocation, teleportAgent, executeCommand]);

  const handleChatMessage = useCallback(async () => {
    if (!selectedAgent || !chatInput.trim()) return;
    
    const message = {
      type: 'user' as const,
      content: chatInput,
      timestamp: Date.now()
    };
    
    setChatMessages(prev => [...prev, message]);
    
    const result = await executeCommand(
      `interactive_chat: ${selectedAgent} - ${chatInput}`,
      () => interactiveChat(selectedAgent, chatInput)
    );
    
    if (result.success) {
      setChatMessages(prev => [...prev, {
        type: 'system',
        content: 'Message sent to agent',
        timestamp: Date.now()
      }]);
    }
    
    setChatInput('');
  }, [selectedAgent, chatInput, interactiveChat, executeCommand]);

  const clearHistory = useCallback(() => {
    setCommandHistory([]);
  }, []);

  const closeModal = useCallback(() => {
    setActiveModal(null);
    setSelectedAgent('');
  }, []);

  const Modal = ({ isOpen, onClose, title, children }: {
    isOpen: boolean;
    onClose: () => void;
    title: string;
    children: React.ReactNode;
  }) => {
    if (!isOpen) return null;

    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
        onClick={onClose}
      >
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.95, opacity: 0 }}
          className="bg-white rounded-lg p-6 w-full max-w-md mx-4 max-h-[90vh] overflow-y-auto"
          onClick={e => e.stopPropagation()}
        >
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-bold text-gray-800">{title}</h2>
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 text-2xl"
            >
              ×
            </button>
          </div>
          {children}
        </motion.div>
      </motion.div>
    );
  };

  const AgentSelector = ({ value, onChange }: { value: string; onChange: (value: string) => void }) => (
    <div className="mb-4">
      <label className="block text-sm font-medium text-gray-700 mb-2">Select Agent:</label>
      <div className="max-h-32 overflow-y-auto border border-gray-300 rounded-lg">
        {simulacraData.activeIds.map(agentId => (
          <div
            key={agentId}
            className={`p-3 cursor-pointer hover:bg-gray-50 border-b border-gray-200 last:border-b-0 ${
              value === agentId ? 'bg-blue-50 border-blue-200' : ''
            }`}
            onClick={() => onChange(agentId)}
          >
            <div className="font-medium text-gray-900">
              {simulacraData.simulacra[agentId]?.persona_details?.Name || agentId}
            </div>
            <div className="text-sm text-gray-600">
              {simulacraData.simulacra[agentId]?.status} - {simulacraData.simulacra[agentId]?.current_action_description}
            </div>
          </div>
        ))}
      </div>
    </div>
  );

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
          
          <button
            onClick={() => setActiveModal('history')}
            className="flex items-center gap-2 px-4 py-3 bg-gray-600 text-white rounded-lg hover:bg-gray-700 transition-colors"
          >
            <History className="w-4 h-4" />
            <span className="text-sm">Command History</span>
          </button>
        </div>

        {/* Recent History Preview */}
        {commandHistory.length > 0 && (
          <div className="mt-6">
            <h3 className="text-sm font-medium text-gray-700 mb-2">Recent Commands</h3>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {commandHistory.slice(0, 3).map(entry => (
                <div key={entry.id} className="text-xs p-2 bg-gray-50 rounded border-l-2 border-gray-300">
                  <div className="font-medium">{entry.command}</div>
                  <div className={`text-xs ${
                    entry.status === 'success' ? 'text-green-600' : 
                    entry.status === 'error' ? 'text-red-600' : 'text-yellow-600'
                  }`}>
                    {entry.status}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Modals */}
      <Modal isOpen={activeModal === 'narrative'} onClose={closeModal} title="Inject Narrative">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Narrative Text:</label>
            <textarea
              value={narrativeText}
              onChange={(e) => setNarrativeText(e.target.value)}
              className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-vertical"
              placeholder="Enter narrative text to inject into the simulation..."
              autoFocus
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={closeModal}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={handleNarrativeInjection}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
            >
              Inject
            </button>
          </div>
        </div>
      </Modal>

      <Modal isOpen={activeModal === 'agent-event'} onClose={closeModal} title="Send Agent Event">
        <div className="space-y-4">
          <AgentSelector value={selectedAgent} onChange={setSelectedAgent} />
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Event Description:</label>
            <textarea
              value={eventDescription}
              onChange={(e) => setEventDescription(e.target.value)}
              className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-vertical"
              placeholder="Describe the event happening to the agent..."
              autoFocus
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={closeModal}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={handleAgentEvent}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
            >
              Send Event
            </button>
          </div>
        </div>
      </Modal>

      <Modal isOpen={activeModal === 'world-info'} onClose={closeModal} title="Update World Info">
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Category:</label>
            <select
              value={worldInfoCategory}
              onChange={(e) => setWorldInfoCategory(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg"
            >
              <option value="weather">Weather</option>
              <option value="news">News</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Information:</label>
            <textarea
              value={worldInfoText}
              onChange={(e) => setWorldInfoText(e.target.value)}
              className="w-full h-32 p-3 border border-gray-300 rounded-lg resize-vertical"
              placeholder="Enter new world information..."
              autoFocus
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={closeModal}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={handleWorldInfoUpdate}
              className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
            >
              Update
            </button>
          </div>
        </div>
      </Modal>

      <Modal isOpen={activeModal === 'teleport'} onClose={closeModal} title="Teleport Agent">
        <div className="space-y-4">
          <AgentSelector value={selectedAgent} onChange={setSelectedAgent} />
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">New Location ID:</label>
            <input
              type="text"
              value={teleportLocation}
              onChange={(e) => setTeleportLocation(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg"
              placeholder="e.g., Home_01, Park_Central"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Location Description (optional):</label>
            <textarea
              value={teleportDescription}
              onChange={(e) => setTeleportDescription(e.target.value)}
              className="w-full h-24 p-3 border border-gray-300 rounded-lg resize-vertical"
              placeholder="Optional description of the new location..."
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={closeModal}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Cancel
            </button>
            <button
              onClick={handleTeleport}
              className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700"
            >
              Teleport
            </button>
          </div>
        </div>
      </Modal>

      <Modal isOpen={activeModal === 'chat'} onClose={closeModal} title="Interactive Chat">
        <div className="space-y-4">
          <AgentSelector value={selectedAgent} onChange={setSelectedAgent} />
          <div className="h-64 border border-gray-300 rounded-lg overflow-y-auto p-3 bg-gray-50">
            {chatMessages.length === 0 ? (
              <div className="text-gray-500 text-center">No messages yet</div>
            ) : (
              <div className="space-y-2">
                {chatMessages.map((msg, index) => (
                  <div
                    key={index}
                    className={`p-2 rounded-lg max-w-[80%] ${
                      msg.type === 'user' 
                        ? 'bg-blue-600 text-white ml-auto text-right' 
                        : msg.type === 'agent'
                        ? 'bg-white border-l-4 border-red-400'
                        : 'bg-yellow-100 border-l-4 border-yellow-400'
                    }`}
                  >
                    <div className="text-sm">{msg.content}</div>
                    <div className="text-xs opacity-70 mt-1">
                      {new Date(msg.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              value={chatInput}
              onChange={(e) => setChatInput(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleChatMessage()}
              className="flex-1 p-3 border border-gray-300 rounded-lg"
              placeholder="Type your message..."
              autoFocus
            />
            <button
              onClick={handleChatMessage}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <div className="flex gap-2">
            <button
              onClick={closeModal}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              End Chat
            </button>
          </div>
        </div>
      </Modal>

      <Modal isOpen={activeModal === 'history'} onClose={closeModal} title="Command History">
        <div className="space-y-4">
          <div className="max-h-96 overflow-y-auto space-y-2">
            {commandHistory.length === 0 ? (
              <div className="text-gray-500 text-center">No commands executed yet</div>
            ) : (
              commandHistory.map(entry => (
                <div
                  key={entry.id}
                  className={`p-3 rounded-lg border-l-4 ${
                    entry.status === 'success' ? 'border-green-500 bg-green-50' :
                    entry.status === 'error' ? 'border-red-500 bg-red-50' :
                    'border-yellow-500 bg-yellow-50'
                  }`}
                >
                  <div className="font-medium text-sm">{entry.command}</div>
                  <div className="text-xs text-gray-600 mt-1">
                    {new Date(entry.timestamp).toLocaleString()}
                  </div>
                  {entry.response && (
                    <div className="text-xs mt-2 font-mono bg-white p-2 rounded">
                      {entry.response}
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
          <div className="flex gap-2">
            <button
              onClick={closeModal}
              className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            >
              Close
            </button>
            <button
              onClick={clearHistory}
              className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
            >
              Clear History
            </button>
          </div>
        </div>
      </Modal>
    </div>
  );
});