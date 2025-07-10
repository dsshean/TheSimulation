import React, { useState, useEffect } from 'react';
import { Activity } from 'lucide-react';

interface LiveStatusIndicatorProps {
  running: boolean;
  simTime: number;
  lastUpdate?: number;
}

export const LiveStatusIndicator: React.FC<LiveStatusIndicatorProps> = ({ 
  running, 
  simTime, 
  lastUpdate 
}) => {
  const [displayTime, setDisplayTime] = useState(simTime);
  const [pulse, setPulse] = useState(false);

  // Update display time every second when running
  useEffect(() => {
    if (!running) return;

    const interval = setInterval(() => {
      setDisplayTime(prev => prev + 1);
      setPulse(true);
      setTimeout(() => setPulse(false), 200);
    }, 1000);

    return () => clearInterval(interval);
  }, [running]);

  // Sync with actual sim time when state updates
  useEffect(() => {
    if (lastUpdate && Math.abs(simTime - displayTime) > 2) {
      setDisplayTime(simTime);
    }
  }, [simTime, lastUpdate, displayTime]);

  const formatTime = (timeInSeconds: number) => {
    const hours = Math.floor(timeInSeconds / 3600);
    const minutes = Math.floor((timeInSeconds % 3600) / 60);
    const seconds = Math.floor(timeInSeconds % 60);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ${seconds}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${seconds}s`;
    } else {
      return `${seconds}s`;
    }
  };

  return (
    <div className="flex items-center gap-2 bg-gray-100 px-3 py-2 rounded-lg">
      <div className={`relative ${pulse ? 'animate-pulse' : ''}`}>
        <Activity 
          className={`w-4 h-4 ${running ? 'text-green-500' : 'text-gray-400'}`}
        />
        {running && (
          <div className="absolute -top-1 -right-1 w-2 h-2 bg-green-400 rounded-full animate-ping"></div>
        )}
      </div>
      
      <div className="text-sm">
        <span className="font-medium text-gray-700">Sim Time:</span>
        <span className={`ml-1 font-mono ${running ? 'text-green-600' : 'text-gray-500'}`}>
          {formatTime(displayTime)}
        </span>
      </div>
      
      <div className={`w-2 h-2 rounded-full ${
        running ? 'bg-green-400 animate-pulse' : 'bg-gray-400'
      }`}></div>
    </div>
  );
};