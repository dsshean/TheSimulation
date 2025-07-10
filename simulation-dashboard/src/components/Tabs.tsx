import React, { useState } from 'react';
import { motion } from 'framer-motion';

interface Tab {
  id: string;
  label: string;
  icon?: React.ReactNode;
  content: React.ReactNode;
}

interface TabsProps {
  tabs: Tab[];
  defaultTab?: string;
  className?: string;
}

export const Tabs: React.FC<TabsProps> = React.memo(({ 
  tabs, 
  defaultTab, 
  className = '' 
}) => {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id);

  const activeTabContent = tabs.find(tab => tab.id === activeTab)?.content;

  return (
    <div className={`flex flex-col h-full ${className}`}>
      <div className="flex border-b border-gray-200 bg-gray-50 rounded-t-lg">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`relative px-4 py-3 text-sm font-medium transition-colors ${
              activeTab === tab.id
                ? 'text-blue-600 border-b-2 border-blue-600 bg-white'
                : 'text-gray-600 hover:text-gray-900 hover:bg-gray-100'
            }`}
          >
            <div className="flex items-center gap-2">
              {tab.icon}
              {tab.label}
            </div>
            {activeTab === tab.id && (
              <motion.div
                layoutId="activeTab"
                className="absolute bottom-0 left-0 right-0 h-0.5 bg-blue-600"
                initial={false}
              />
            )}
          </button>
        ))}
      </div>
      
      <div className="flex-1 overflow-hidden">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
          className="h-full"
        >
          {activeTabContent}
        </motion.div>
      </div>
    </div>
  );
}, (prevProps, nextProps) => {
  // Only re-render if tab structure has changed
  // Don't re-render for content updates - let the individual tab content handle that
  return (
    prevProps.tabs.length === nextProps.tabs.length &&
    prevProps.defaultTab === nextProps.defaultTab &&
    prevProps.className === nextProps.className &&
    JSON.stringify(prevProps.tabs.map(t => ({ id: t.id, label: t.label }))) === 
    JSON.stringify(nextProps.tabs.map(t => ({ id: t.id, label: t.label })))
  );
});