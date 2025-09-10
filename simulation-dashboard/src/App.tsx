import React, { useState } from 'react';
import { useTauriSimulation } from './hooks/useTauriSimulation';
import { WorldState } from './components/WorldState';
import { AgentLocationStatus } from './components/AgentLocationStatus';
import { LastGeneratedImage } from './components/LastGeneratedImage';
import { EventLog } from './components/EventLog';
import { EventLogBox } from './components/EventLogBox';
import { LoadingSpinner } from './components/LoadingSpinner';
import { Tabs } from './components/Tabs';
import { D3Visualization } from './components/D3Visualization';
import { SimpleInteractiveCommands } from './components/SimpleInteractiveCommands';
import { LiveStatusIndicator } from './components/LiveStatusIndicator';
import { LiveEventFeed } from './components/LiveEventFeed';
import { Users, MapPin, Activity, RefreshCw, BarChart3, Terminal, Eye, Brain, Zap, BookOpen } from 'lucide-react';

function App() {
  const { 
    state, 
    events, 
    loading, 
    error, 
    running,
    startSimulation,
    stopSimulation,
    refreshState,
    setInputFocus,
    injectNarrative,
    sendAgentEvent,
    updateWorldInfo,
    teleportAgent,
    interactiveChat
  } = useTauriSimulation();
  const [activeView, setActiveView] = useState<'dashboard' | 'visualization' | 'interactive'>('dashboard');
  const [selectedAgent, setSelectedAgent] = useState<string>('all');

  if (loading && !state) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="lg" message="Loading simulation data..." />
          <p className="mt-4 text-gray-600">
            Running: {running ? 'Yes' : 'No'} | Error: {error || 'None'}
          </p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-600 text-6xl mb-4">⚠️</div>
          <h1 className="text-2xl font-bold text-gray-800 mb-4">Error Loading Simulation</h1>
          <p className="text-gray-600 mb-4">{error}</p>
          <p className="text-sm text-gray-500 mb-4">
            Is the simulation running? Try: python main_async.py
          </p>
          <button
            onClick={refreshState}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <RefreshCw className="w-4 h-4 inline mr-2" />
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!state) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="lg" message="Connecting to simulation..." />
          <p className="mt-4 text-gray-600">
            Running: {running ? 'Yes' : 'No'} | Error: {error || 'None'}
          </p>
          <p className="text-sm text-gray-500 mt-2">
            The dashboard will automatically load when simulation data is available
          </p>
        </div>
      </div>
    );
  }

  // Get available agents from state
  const availableAgents = (() => {
    if (!state?.simulacra_profiles) return [];
    return Object.entries(state.simulacra_profiles).map(([id, agent]: [string, any]) => ({
      id,
      name: agent.persona_details?.Name || id
    }));
  })();

  // Filter events by selected agent
  const filteredEvents = (() => {
    if (selectedAgent === 'all') return events;
    const selectedAgentData = availableAgents.find(agent => agent.id === selectedAgent);
    const agentName = selectedAgentData?.name || selectedAgent;
    return events.filter(event => 
      event.agent_id === selectedAgent || 
      event.agent_id === agentName
    );
  })();

  // Use the new LiveEventFeed for better event display
  const eventTabs = [
    {
      id: 'all',
      label: 'Live Feed',
      icon: <Activity className="w-4 h-4" />,
      content: <LiveEventFeed events={events} maxEvents={10} />
    },
    {
      id: 'simulacra',
      label: 'Simulacra',
      icon: <Users className="w-4 h-4" />,
      content: <EventLog events={events.filter(e => e.event_type === 'monologue' || e.event_type === 'intent' || e.event_type === 'observation')} title="Simulacra Events" />
    },
    {
      id: 'world',
      label: 'World Engine',
      icon: <MapPin className="w-4 h-4" />,
      content: <EventLog events={events.filter(e => e.event_type === 'world_engine')} title="World Engine Events" />
    },
    {
      id: 'narrative',
      label: 'Narrative',
      icon: <BookOpen className="w-4 h-4" />,
      content: <EventLog events={events.filter(e => e.event_type === 'narrative')} title="Narrative Events" />
    }
  ];

  return (
    <div className="min-h-screen bg-gray-100">
      <div className="p-6">
        <div className="max-w-full mx-auto px-4">
          {/* Header */}
          <div className="mb-6 flex items-center justify-between">
            <div>
              <h1 className="text-3xl font-bold text-gray-800 mb-2">
                TheSimulation Dashboard
              </h1>
              <p className="text-gray-600">
                Real-time monitoring of autonomous agent simulation
              </p>
            </div>
            <div className="flex items-center gap-4">
              <LiveStatusIndicator 
                running={running} 
                simTime={state?.world_time || 0} 
                lastUpdate={Date.now()}
              />
              
              {/* Agent Selector */}
              {availableAgents.length > 0 && (
                <div className="flex items-center gap-2">
                  <label className="text-sm font-medium text-gray-700">Agent:</label>
                  <select
                    value={selectedAgent}
                    onChange={(e) => setSelectedAgent(e.target.value)}
                    className="px-3 py-1 text-sm border border-gray-300 rounded-md bg-white focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="all">All Agents</option>
                    {availableAgents.map((agent) => (
                      <option key={agent.id} value={agent.id}>
                        {agent.name}
                      </option>
                    ))}
                  </select>
                </div>
              )}
              
              {/* View Toggle */}
              <div className="flex bg-gray-200 rounded-lg p-1">
                <button
                  onClick={() => setActiveView('dashboard')}
                  className={`px-3 py-2 rounded-md text-sm flex items-center gap-2 transition-colors ${
                    activeView === 'dashboard' 
                      ? 'bg-white text-blue-600 shadow-sm' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Eye className="w-4 h-4" />
                  Dashboard
                </button>
                <button
                  onClick={() => setActiveView('visualization')}
                  className={`px-3 py-2 rounded-md text-sm flex items-center gap-2 transition-colors ${
                    activeView === 'visualization' 
                      ? 'bg-white text-blue-600 shadow-sm' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <BarChart3 className="w-4 h-4" />
                  Visualization
                </button>
                <button
                  onClick={() => setActiveView('interactive')}
                  className={`px-3 py-2 rounded-md text-sm flex items-center gap-2 transition-colors ${
                    activeView === 'interactive' 
                      ? 'bg-white text-blue-600 shadow-sm' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Terminal className="w-4 h-4" />
                  Interactive
                </button>
              </div>
              
              <button
                onClick={refreshState}
                className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                Refresh
              </button>
            </div>
          </div>

          {/* Conditional Content Based on Active View */}
          {activeView === 'dashboard' && (
            <>
              {/* Main Grid Layout - Resizable */}
              <div className="flex gap-6 mb-6">
                {/* Left Column - World State and Simulation Outputs */}
                <div className="flex-1 min-w-0 space-y-6">
                  <WorldState state={state} />
                  
                  {/* Simulation Outputs */}
                  <Tabs
                    tabs={[
                      {
                        id: 'main-events',
                        label: 'Simulacra & Narrative',
                        icon: <Brain className="w-4 h-4" />,
                        content: (
                          <div className="space-y-6">
                            {/* Simulacra Thoughts and Observations */}
                            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                              <EventLogBox
                                title={selectedAgent === 'all' ? 'Simulacra Thoughts' : `${availableAgents.find(a => a.id === selectedAgent)?.name || selectedAgent} Thoughts`}
                                events={filteredEvents.filter(e => e.event_type === 'monologue')}
                                icon={<Brain className="w-5 h-5" />}
                                bgColor="bg-blue-50"
                                textColor="text-blue-900"
                              />
                              <EventLogBox
                                title={selectedAgent === 'all' ? 'Simulacra Observations' : `${availableAgents.find(a => a.id === selectedAgent)?.name || selectedAgent} Observations`}
                                events={filteredEvents.filter(e => e.event_type === 'observation')}
                                icon={<Eye className="w-5 h-5" />}
                                bgColor="bg-cyan-50"
                                textColor="text-cyan-900"
                              />
                            </div>
                            
                            {/* Narrative */}
                            <div>
                              <EventLogBox
                                title="Narrative"
                                events={filteredEvents.filter(e => e.event_type === 'narrative')}
                                icon={<BookOpen className="w-5 h-5" />}
                                bgColor="bg-orange-50"
                                textColor="text-orange-900"
                              />
                            </div>
                          </div>
                        )
                      },
                      {
                        id: 'world-engine',
                        label: 'World Engine',
                        icon: <Zap className="w-4 h-4" />,
                        content: (
                          <div className="bg-green-50 rounded-lg p-4 max-h-96 overflow-y-auto">
                            <EventLog 
                              events={filteredEvents.filter(e => e.event_type === 'world_engine')} 
                              title="World Engine Events"
                            />
                          </div>
                        )
                      }
                    ]}
                  />
                </div>
                
                {/* Right Column - Agent Status and Image */}
                <div style={{width: '500px'}} className="flex-shrink-0 space-y-6">
                  <AgentLocationStatus state={state} />
                  {/* Last Generated Image */}
                  <div className="bg-white rounded-lg shadow-lg border border-gray-200">
                    <div className="p-4 border-b border-gray-200">
                      <h3 className="text-lg font-bold text-gray-800 flex items-center gap-2">
                        🖼️ Latest Generated Image
                      </h3>
                    </div>
                    <div className="p-4">
                      <LastGeneratedImage state={state} />
                    </div>
                  </div>
                </div>
              </div>

            </>
          )}

          {activeView === 'visualization' && (
            <div className="h-[70vh] bg-white rounded-lg shadow-lg border border-gray-200">
              <D3Visualization 
                state={state} 
                width={1200} 
                height={600} 
              />
            </div>
          )}

          {activeView === 'interactive' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <SimpleInteractiveCommands 
                state={state}
                onInjectNarrative={injectNarrative}
                onSendAgentEvent={sendAgentEvent}
                onUpdateWorldInfo={updateWorldInfo}
                onTeleportAgent={teleportAgent}
                onInteractiveChat={interactiveChat}
                setInputFocus={setInputFocus}
              />
              <LiveEventFeed events={events} maxEvents={10} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;