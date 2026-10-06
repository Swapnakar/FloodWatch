import React, { useState } from 'react';
import { Target, MapPin, Sliders, Activity, Menu, Droplets } from 'lucide-react';
import './Sidebar.css';

export default function Sidebar({ activeTab, setActiveTab }) {
  const [isExpanded, setIsExpanded] = useState(false);

  const tabs = [
    { id: 'prediction', label: 'Prediction', icon: <Target size={20} /> },
    { id: 'route', label: 'Safe Route', icon: <MapPin size={20} /> },
    { id: 'simulation', label: 'Simulation', icon: <Sliders size={20} /> },
    { id: 'drainage', label: 'Drainage', icon: <Droplets size={20} /> },
    { id: 'connectivity', label: 'Connectivity', icon: <Activity size={20} /> },
  ];

  return (
    <aside className={`app-sidebar ${isExpanded ? 'expanded' : 'collapsed'}`}>
      <div className="sidebar-header">
        <button className="sidebar-toggle" onClick={() => setIsExpanded(!isExpanded)}>
          <Menu size={24} color="var(--ink)" />
        </button>
        {isExpanded && <div className="sidebar-title">FLOODEXA</div>}
      </div>
      
      <nav className="sidebar-nav">
        {tabs.map(tab => (
          <button 
            key={tab.id}
            className={`sidebar-tab ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
            title={tab.label}
          >
            <div className="sidebar-icon">{tab.icon}</div>
            {isExpanded && <span className="sidebar-label">{tab.label}</span>}
          </button>
        ))}
      </nav>
    </aside>
  );
}
