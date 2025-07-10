import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { SimulationState } from '../types/simulation';

interface Node {
  id: string;
  type: 'location' | 'agent' | 'object';
  name: string;
  description?: string;
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
}

interface Link {
  source: string | Node;
  target: string | Node;
  type: 'connection' | 'agent_at_location';
}

interface D3VisualizationProps {
  state: SimulationState;
  width?: number;
  height?: number;
}

export const D3Visualization: React.FC<D3VisualizationProps> = React.memo(({
  state,
  width = 800,
  height = 600
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const simulationRef = useRef<d3.Simulation<Node, Link> | null>(null);
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [tooltip, setTooltip] = useState<{ x: number; y: number; content: string } | null>(null);

  // Create a stable data representation that only changes when visualization-relevant data changes
  const visualizationData = useMemo(() => {
    if (!state) return null;
    
    return {
      locations: state.current_world_state?.location_details || {},
      agents: state.simulacra_profiles || {},
      activeAgents: state.active_simulacra_ids || [],
      // Only include time if it's used for positioning (rounded to reduce updates)
      timeSlice: Math.floor((state.world_time || 0) / 30) // Only update every 30 time units
    };
  }, [
    state?.current_world_state?.location_details,
    state?.simulacra_profiles,
    state?.active_simulacra_ids,
    Math.floor((state?.world_time || 0) / 30)
  ]);

  useEffect(() => {
    if (!svgRef.current || !visualizationData) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    // Create groups for different elements
    const linksGroup = svg.append('g').attr('class', 'links');
    const nodesGroup = svg.append('g').attr('class', 'nodes');
    const labelsGroup = svg.append('g').attr('class', 'labels');

    // Process simulation data into nodes and links
    const nodes: Node[] = [];
    const links: Link[] = [];

    // Add location nodes
    const locationDetails = visualizationData.locations;
    Object.entries(locationDetails).forEach(([locationId, location]) => {
      nodes.push({
        id: locationId,
        type: 'location',
        name: location.name || locationId,
        description: location.description
      });

      // Add connections between locations
      location.connected_locations?.forEach(conn => {
        const targetId = conn.to_location_id_hint;
        if (targetId && locationDetails[targetId]) {
          links.push({
            source: locationId,
            target: targetId,
            type: 'connection'
          });
        }
      });

      // Don't add individual objects as separate nodes anymore
      // We'll show object counts inside location boxes instead
    });

    // Add agent nodes
    const simulacra = visualizationData.agents;
    const activeIds = visualizationData.activeAgents;
    
    activeIds.forEach(agentId => {
      const agent = simulacra[agentId];
      if (agent) {
        nodes.push({
          id: agentId,
          type: 'agent',
          name: agent.persona_details?.Name || agentId,
          description: `${agent.status} - ${agent.current_action_description}`
        });

        // Link agent to their current location
        if (agent.current_location && locationDetails[agent.current_location]) {
          links.push({
            source: agentId,
            target: agent.current_location,
            type: 'agent_at_location'
          });
        }
      }
    });

    // Create force simulation
    const simulation = d3.forceSimulation<Node>(nodes)
      .force('link', d3.forceLink<Node, Link>(links).id(d => d.id).distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(d => getNodeRadius(d) + 5));

    simulationRef.current = simulation;

    // Create links
    const link = linksGroup
      .selectAll('line')
      .data(links)
      .enter()
      .append('line')
      .attr('class', d => `link link-${d.type}`)
      .attr('stroke', d => d.type === 'agent_at_location' ? '#10b981' : '#adb5bd')
      .attr('stroke-width', d => d.type === 'agent_at_location' ? 3 : 1.5)
      .attr('stroke-dasharray', d => d.type === 'agent_at_location' ? '6,6' : 'none');

    // Create nodes
    const node = nodesGroup
      .selectAll('circle')
      .data(nodes)
      .enter()
      .append('circle')
      .attr('class', d => `node node-${d.type}`)
      .attr('r', getNodeRadius)
      .attr('fill', getNodeColor)
      .attr('stroke', '#444')
      .attr('stroke-width', 1.5)
      .style('cursor', 'pointer')
      .call(d3.drag<SVGCircleElement, Node>()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended)
      )
      .on('click', (event, d) => {
        setSelectedNode(d);
      })
      .on('mouseover', (event, d) => {
        const [x, y] = d3.pointer(event, svg.node());
        setTooltip({
          x,
          y,
          content: `${d.name}\n${d.description || ''}`
        });
      })
      .on('mouseout', () => {
        setTooltip(null);
      });

    // Create labels with object/NPC counts for locations
    const labelGroups = labelsGroup
      .selectAll('g')
      .data(nodes)
      .enter()
      .append('g')
      .attr('class', 'node-label-group')
      .attr('text-anchor', 'middle')
      .attr('pointer-events', 'none');

    // Add location name
    labelGroups.append('text')
      .attr('class', 'node-text-name')
      .attr('dy', d => d.type === 'location' ? '-0.3em' : '.35em')
      .attr('font-size', '11px')
      .attr('font-weight', d => d.type === 'location' ? 'bold' : 'normal')
      .attr('fill', '#343a40')
      .text(d => d.name.length > 12 ? d.name.substring(0, 12) + '...' : d.name);

    // Add object/NPC counts for locations only
    labelGroups.filter(d => d.type === 'location')
      .append('text')
      .attr('class', 'node-text-counts')
      .attr('dy', '0.8em')
      .attr('font-size', '9px')
      .attr('fill', '#666')
      .text(d => {
        const location = visualizationData.locations[d.id];
        const objectCount = location?.ephemeral_objects?.length || 0;
        const npcCount = location?.npcs?.length || 0;
        return `${objectCount} objects / ${npcCount} npcs`;
      });

    // Update positions on tick
    simulation.on('tick', () => {
      link
        .attr('x1', d => (d.source as Node).x!)
        .attr('y1', d => (d.source as Node).y!)
        .attr('x2', d => (d.target as Node).x!)
        .attr('y2', d => (d.target as Node).y!);

      node
        .attr('cx', d => d.x!)
        .attr('cy', d => d.y!);

      labelGroups
        .attr('transform', d => `translate(${d.x!},${d.y!})`);
    });

    // Drag functions
    function dragstarted(event: any, d: Node) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      d.fx = d.x;
      d.fy = d.y;
    }

    function dragged(event: any, d: Node) {
      d.fx = event.x;
      d.fy = event.y;
    }

    function dragended(event: any, d: Node) {
      if (!event.active) simulation.alphaTarget(0);
      d.fx = null;
      d.fy = null;
    }

    // Zoom behavior
    const zoom = d3.zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.1, 4])
      .on('zoom', (event) => {
        svg.selectAll('g').attr('transform', event.transform);
      });

    svg.call(zoom);

    return () => {
      if (simulationRef.current) {
        simulationRef.current.stop();
      }
    };
  }, [visualizationData, width, height]);

  function getNodeRadius(d: Node): number {
    switch (d.type) {
      case 'location': return 20;
      case 'agent': return 15;
      case 'object': return 8;
      default: return 10;
    }
  }

  function getNodeColor(d: Node): string {
    switch (d.type) {
      case 'location': return '#3b82f6'; // Nice blue
      case 'agent': return '#10b981'; // Nice green
      case 'object': return '#4ecdc4';
      default: return '#999';
    }
  }

  const resetView = () => {
    const svg = d3.select(svgRef.current);
    svg.transition().duration(750).call(
      d3.zoom<SVGSVGElement, unknown>().transform,
      d3.zoomIdentity
    );
  };

  const centerOnAgents = () => {
    if (!simulationRef.current) return;
    
    const agentNodes = simulationRef.current.nodes().filter(n => n.type === 'agent');
    if (agentNodes.length === 0) return;

    const centerX = d3.mean(agentNodes, d => d.x!) || width / 2;
    const centerY = d3.mean(agentNodes, d => d.y!) || height / 2;

    const svg = d3.select(svgRef.current);
    svg.transition().duration(750).call(
      d3.zoom<SVGSVGElement, unknown>().transform,
      d3.zoomIdentity.translate(width / 2 - centerX, height / 2 - centerY)
    );
  };

  return (
    <div className="relative w-full h-full bg-gray-50">
      {/* Controls */}
      <div className="absolute top-4 left-4 z-10 flex gap-2">
        <button
          onClick={resetView}
          className="px-3 py-2 bg-white border border-gray-300 rounded-lg text-sm hover:bg-gray-50 shadow-sm"
        >
          Reset View
        </button>
        <button
          onClick={centerOnAgents}
          className="px-3 py-2 bg-white border border-gray-300 rounded-lg text-sm hover:bg-gray-50 shadow-sm"
        >
          Center on Agents
        </button>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 left-4 z-10 bg-white p-4 rounded-lg shadow-lg border border-gray-200">
        <div className="text-sm font-medium mb-2">Legend</div>
        <div className="space-y-2 text-xs">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#3b82f6]"></div>
            <span>Locations (with object/NPC counts)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-[#10b981]"></div>
            <span>Agents</span>
          </div>
        </div>
      </div>

      {/* SVG Visualization */}
      <svg
        ref={svgRef}
        width={width}
        height={height}
        className="w-full h-full"
      />

      {/* Tooltip */}
      {tooltip && (
        <div
          className="absolute pointer-events-none bg-white p-3 rounded-lg shadow-lg border border-gray-200 text-sm max-w-xs z-20"
          style={{
            left: tooltip.x + 10,
            top: tooltip.y - 10,
          }}
        >
          <pre className="whitespace-pre-wrap">{tooltip.content}</pre>
        </div>
      )}

      {/* Selected Node Info */}
      {selectedNode && (
        <div className="absolute top-4 right-4 z-10 bg-white p-4 rounded-lg shadow-lg border border-gray-200 max-w-sm">
          <div className="flex justify-between items-start mb-2">
            <h3 className="font-medium text-lg">{selectedNode.name}</h3>
            <button
              onClick={() => setSelectedNode(null)}
              className="text-gray-400 hover:text-gray-600"
            >
              ×
            </button>
          </div>
          <div className="text-sm text-gray-600">
            <div className="mb-1">Type: {selectedNode.type}</div>
            {selectedNode.description && (
              <div>Description: {selectedNode.description}</div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}, (prevProps, nextProps) => {
  // Only re-render if visualization-relevant data has changed
  if (!prevProps.state || !nextProps.state) return false;
  
  return (
    JSON.stringify(prevProps.state.current_world_state?.location_details) === 
    JSON.stringify(nextProps.state.current_world_state?.location_details) &&
    JSON.stringify(prevProps.state.simulacra_profiles) === 
    JSON.stringify(nextProps.state.simulacra_profiles) &&
    JSON.stringify(prevProps.state.active_simulacra_ids) === 
    JSON.stringify(nextProps.state.active_simulacra_ids) &&
    prevProps.width === nextProps.width &&
    prevProps.height === nextProps.height &&
    Math.floor((prevProps.state.world_time || 0) / 30) === Math.floor((nextProps.state.world_time || 0) / 30)
  );
});