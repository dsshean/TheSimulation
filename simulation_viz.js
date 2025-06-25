// TheSimulation Web Visualizer
// Real-time D3.js visualization with WebSocket connection

class SimulationVisualizer {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.isPaused = false;
        this.currentData = null;
        
        // D3 setup
        this.svg = d3.select("#visualization");
        this.width = window.innerWidth - 350; // Subtract info panel width
        this.height = window.innerHeight - 50; // Subtract status bar height
        
        this.svg
            .attr("width", this.width)
            .attr("height", this.height);
        
        // Create groups for different elements
        this.linksGroup = this.svg.append("g").attr("class", "links");
        this.nodesGroup = this.svg.append("g").attr("class", "nodes");
        this.labelsGroup = this.svg.append("g").attr("class", "labels");
        
        // Force simulation setup (t-SNE-like layout with better stability)
        this.simulation = d3.forceSimulation()
            .force("link", d3.forceLink().id(d => d.id).distance(this.calculateLinkDistance))
            .force("charge", d3.forceManyBody().strength(-200))
            .force("center", d3.forceCenter(this.width / 2, this.height / 2))
            .force("collision", d3.forceCollide().radius(d => this.calculateNodeRadius(d) + 5))
            .force("x", d3.forceX(this.width / 2).strength(0.05))
            .force("agentAttraction", this.createAgentAttractionForce())  // Keep agents near their locations
            .alphaDecay(0.02)  // Faster settling
            .velocityDecay(0.8)  // More damping to reduce bouncing
            .force("y", d3.forceY(this.height / 2).strength(0.1));
        
        // Zoom and pan behavior
        this.zoom = d3.zoom()
            .scaleExtent([0.1, 4])
            .on("zoom", (event) => {
                this.svg.selectAll("g").attr("transform", event.transform);
            });
        
        this.svg.call(this.zoom);
        
        // Tooltip
        this.tooltip = d3.select("#tooltip");
        
        // Initialize connection
        this.connect();
        
        // Update real time clock
        this.updateClock();
        setInterval(() => this.updateClock(), 1000);
        
        // Handle window resize
        window.addEventListener('resize', () => this.handleResize());
    }
    
    connect() {
        try {
            this.websocket = new WebSocket('ws://localhost:8766');
            
            this.websocket.onopen = () => {
                console.log('Connected to simulation');
                this.isConnected = true;
                this.updateConnectionStatus(true);
                document.getElementById('loadingIndicator').style.display = 'none';
            };
            
            this.websocket.onmessage = (event) => {
                if (!this.isPaused) {
                    const data = JSON.parse(event.data);
                    this.handleSimulationUpdate(data);
                }
            };
            
            this.websocket.onclose = () => {
                console.log('Disconnected from simulation');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                
                // Attempt to reconnect after 3 seconds
                setTimeout(() => {
                    if (!this.isConnected) {
                        console.log('Attempting to reconnect...');
                        this.connect();
                    }
                }, 3000);
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.updateConnectionStatus(false);
            };
            
        } catch (error) {
            console.error('Failed to connect:', error);
            this.updateConnectionStatus(false);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusDot = document.getElementById('connectionStatus');
        const statusText = document.getElementById('connectionText');
        
        if (connected) {
            statusDot.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            statusDot.classList.remove('connected');
            statusText.textContent = 'Disconnected';
        }
    }
    
    handleSimulationUpdate(data) {
        this.currentData = data;
        
        // Update simulation time
        document.getElementById('simTime').textContent = `Sim Time: ${Math.round(data.world_time)}s`;
        
        // Update visualization
        this.updateVisualization(data);
        
        // Update info panels
        this.updateAgentsList(data.simulacra);
        this.updateLocationsList(data.locations);
        this.updateNarrativeList(data.recent_narrative);
        this.updateWorldStatus(data);
    }
    
    updateVisualization(data) {
        // Prepare nodes and links
        const nodes = [];
        const links = [];
        
        // Add location nodes
        Object.values(data.locations).forEach(location => {
            nodes.push({
                id: location.id,
                type: 'location',
                name: location.name,
                data: location,
                x: location.x || Math.random() * this.width,
                y: location.y || Math.random() * this.height
            });
        });
        
        // Add agent nodes and links to their current locations
        Object.values(data.simulacra).forEach(agent => {
            // If agent has a current location, position them near/inside it
            let agentX = Math.random() * this.width;
            let agentY = Math.random() * this.height;
            
            // Try to position agent near their current location if it exists
            if (agent.current_location && data.locations[agent.current_location]) {
                const location = data.locations[agent.current_location];
                if (location.x !== undefined && location.y !== undefined) {
                    // Position agent slightly offset from location center
                    agentX = location.x + (Math.random() - 0.5) * 40;
                    agentY = location.y + (Math.random() - 0.5) * 40;
                }
            }
            
            nodes.push({
                id: agent.id,
                type: 'agent',
                name: agent.name,
                data: agent,
                x: agent.x || agentX,
                y: agent.y || agentY
            });
            
            // Link agent to current location with dotted line
            if (agent.current_location && data.locations[agent.current_location]) {
                links.push({
                    source: agent.id,
                    target: agent.current_location,
                    type: 'agent-location'
                });
            }
        });
        
        // Add location connections
        Object.values(data.locations).forEach(location => {
            if (location.connections && Array.isArray(location.connections)) {
                location.connections.forEach(connection => {
                    // Handle both string IDs and connection objects
                    const targetId = typeof connection === 'string' 
                        ? connection 
                        : connection.to_location_id_hint || connection.target_id;
                    
                    if (targetId && data.locations[targetId]) {
                        // Avoid duplicate bidirectional links
                        const linkId = location.id < targetId 
                            ? `${location.id}-${targetId}` 
                            : `${targetId}-${location.id}`;
                        
                        // Check if this link already exists
                        if (!links.some(link => 
                            `${link.source}-${link.target}` === linkId || 
                            `${link.target}-${link.source}` === linkId
                        )) {
                            links.push({
                                source: location.id,
                                target: targetId,
                                type: 'location-connection'
                            });
                        }
                    }
                });
            }
        });
        
        // Update D3 visualization
        this.updateD3Visualization(nodes, links);
    }
    
    updateD3Visualization(nodes, links) {
        // Update links
        const link = this.linksGroup
            .selectAll(".link")
            .data(links, d => `${d.source.id || d.source}-${d.target.id || d.target}`);
        
        link.exit().remove();
        
        const linkEnter = link.enter()
            .append("line")
            .attr("class", d => `link ${d.type}`);
        
        link.merge(linkEnter)
            .attr("class", d => `link ${d.type}`)
            .style("stroke", d => {
                switch(d.type) {
                    case 'agent-location': return '#ff6b6b';
                    case 'location-connection': return 'rgba(255, 255, 255, 0.4)';
                    default: return 'rgba(255, 255, 255, 0.2)';
                }
            })
            .style("stroke-width", d => d.type === 'agent-location' ? 3 : 1)
            .style("stroke-dasharray", d => d.type === 'agent-location' ? "8,4" : "none")
            .style("opacity", d => d.type === 'agent-location' ? 0.8 : 0.6);
        
        // Update nodes
        const node = this.nodesGroup
            .selectAll(".node")
            .data(nodes, d => d.id);
        
        node.exit().remove();
        
        const nodeEnter = node.enter()
            .append("circle")
            .attr("class", d => `node node-${d.type}`)
            .attr("r", this.calculateNodeRadius)
            .style("fill", d => this.getNodeColor(d))
            .style("stroke", "#ffffff")
            .style("stroke-width", d => d.type === 'agent' ? 2 : 1)
            .on("mouseover", (event, d) => this.showTooltip(event, d))
            .on("mouseout", () => this.hideTooltip())
            .on("click", (event, d) => this.handleNodeClick(event, d))
            .call(d3.drag()
                .on("start", (event, d) => this.dragStarted(event, d))
                .on("drag", (event, d) => this.dragged(event, d))
                .on("end", (event, d) => this.dragEnded(event, d)));
        
        node.merge(nodeEnter)
            .transition()
            .duration(300)
            .attr("r", this.calculateNodeRadius)
            .style("fill", d => this.getNodeColor(d));
        
        // Update labels
        const label = this.labelsGroup
            .selectAll(".node-text")
            .data(nodes, d => d.id);
        
        label.exit().remove();
        
        const labelEnter = label.enter()
            .append("text")
            .attr("class", "node-text")
            .style("font-size", d => d.type === 'location' ? "12px" : "10px")
            .style("font-weight", d => d.type === 'agent' ? "bold" : "normal");
        
        label.merge(labelEnter)
            .text(d => d.name)
            .attr("dy", d => this.calculateNodeRadius(d) + 15);
        
        // Update simulation
        this.simulation.nodes(nodes);
        this.simulation.force("link").links(links);
        
        // Only restart with gentle animation if significant changes occurred
        if (this.shouldRestartSimulation(nodes, links)) {
            this.simulation.alpha(0.2).restart();
            
            // Set a timeout to stop the simulation after it settles
            clearTimeout(this.settlementTimeout);
            this.settlementTimeout = setTimeout(() => {
                if (this.simulation.alpha() < 0.01) {
                    this.simulation.stop();
                    console.log("Simulation stopped - settled into stable positions");
                }
            }, 3000);
        } else {
            // Don't restart physics for minor updates - just update visual properties
            // This prevents the constant bouncing on every data refresh
        }
        
        // Update positions on tick with auto-stop mechanism
        this.simulation.on("tick", () => {
            this.linksGroup.selectAll(".link")
                .attr("x1", d => d.source.x)
                .attr("y1", d => d.source.y)
                .attr("x2", d => d.target.x)
                .attr("y2", d => d.target.y);
            
            this.nodesGroup.selectAll(".node")
                .attr("cx", d => d.x)
                .attr("cy", d => d.y);
            
            this.labelsGroup.selectAll(".node-text")
                .attr("x", d => d.x)
                .attr("y", d => d.y);
        });
        
        // Auto-stop simulation when it settles
        this.simulation.on("end", () => {
            console.log("Simulation settled");
        });
    }
    
    shouldRestartSimulation(nodes, links) {
        // Check if this is the first time
        if (!this.lastNodes || !this.lastLinks) {
            this.lastNodes = [...nodes];
            this.lastLinks = [...links];
            return true;
        }
        
        // Check for structural changes (node/link count changes)
        if (nodes.length !== this.lastNodes.length || links.length !== this.lastLinks.length) {
            this.lastNodes = [...nodes];
            this.lastLinks = [...links];
            return true;
        }
        
        // Check for new nodes or removed nodes (by ID)
        const currentNodeIds = new Set(nodes.map(n => n.id));
        const lastNodeIds = new Set(this.lastNodes.map(n => n.id));
        
        if (currentNodeIds.size !== lastNodeIds.size || 
            ![...currentNodeIds].every(id => lastNodeIds.has(id))) {
            this.lastNodes = [...nodes];
            this.lastLinks = [...links];
            return true;
        }
        
        // Check for new links or removed links
        const currentLinkIds = new Set(links.map(l => `${l.source}-${l.target}`));
        const lastLinkIds = new Set(this.lastLinks.map(l => `${l.source}-${l.target}`));
        
        if (currentLinkIds.size !== lastLinkIds.size || 
            ![...currentLinkIds].every(id => lastLinkIds.has(id))) {
            this.lastNodes = [...nodes];
            this.lastLinks = [...links];
            return true;
        }
        
        // For everything else (status changes, position updates), just update data without restarting
        this.updateNodeDataOnly(nodes);
        return false;
    }
    
    updateNodeDataOnly(nodes) {
        // Preserve existing positions for stable nodes
        if (this.lastNodes) {
            nodes.forEach(node => {
                const lastNode = this.lastNodes.find(n => n.id === node.id);
                if (lastNode && lastNode.x !== undefined && lastNode.y !== undefined) {
                    node.x = lastNode.x;
                    node.y = lastNode.y;
                    node.fx = lastNode.fx;
                    node.fy = lastNode.fy;
                }
            });
        }
        
        // Update node colors and properties without restarting simulation
        this.nodesGroup.selectAll(".node")
            .data(nodes, d => d.id)
            .style("fill", d => this.getNodeColor(d));
        
        // Update labels
        this.labelsGroup.selectAll(".node-text")
            .data(nodes, d => d.id)
            .text(d => d.name);
        
        // Store updated nodes for position preservation
        this.lastNodes = [...nodes];
    }
    
    createAgentAttractionForce() {
        return () => {
            // This force attracts agents toward their current location
            this.simulation.nodes().forEach(node => {
                if (node.type === 'agent' && node.data && node.data.current_location) {
                    // Find the location node
                    const locationNode = this.simulation.nodes().find(n => 
                        n.type === 'location' && n.id === node.data.current_location
                    );
                    
                    if (locationNode) {
                        // Apply gentle attraction toward location
                        const dx = locationNode.x - node.x;
                        const dy = locationNode.y - node.y;
                        const distance = Math.sqrt(dx * dx + dy * dy);
                        
                        if (distance > 30) { // Only attract if agent is far from location
                            const strength = 0.1;
                            node.vx += dx * strength * 0.1;
                            node.vy += dy * strength * 0.1;
                        }
                    }
                }
            });
        };
    }
    
    calculateNodeRadius(d) {
        switch(d.type) {
            case 'location': return 20 + (d.data?.object_count || 0) * 2;
            case 'agent': return 15;
            case 'object': return 8;
            default: return 10;
        }
    }
    
    calculateLinkDistance(d) {
        if (d.type === 'agent-location') return 80;
        if (d.type === 'location-connection') return 150;
        return 100;
    }
    
    getNodeColor(d) {
        switch(d.type) {
            case 'location': return '#69b3a2';
            case 'agent': 
                switch(d.data?.status) {
                    case 'busy': return '#ff6b6b';
                    case 'idle': return '#4CAF50';
                    case 'thinking': return '#FFA726';
                    default: return '#ff6b6b';
                }
            case 'object': return '#4ecdc4';
            default: return '#cccccc';
        }
    }
    
    showTooltip(event, d) {
        let content = `<strong>${d.name}</strong><br>`;
        content += `Type: ${d.type}<br>`;
        
        if (d.type === 'agent') {
            content += `Status: <span style="color: ${this.getNodeColor(d)}">${d.data.status}</span><br>`;
            content += `<strong>Current Location:</strong> ${d.data.current_location}<br>`;
            content += `Action: ${d.data.current_action || 'None'}<br>`;
            if (d.data.last_observation) {
                content += `Last Seen: ${d.data.last_observation.substring(0, 100)}...<br>`;
            }
            if (d.data.goal && d.data.goal !== 'No current goal') {
                content += `Goal: ${d.data.goal.substring(0, 80)}...<br>`;
            }
        } else if (d.type === 'location') {
            content += `Objects: ${d.data.object_count}<br>`;
            content += `NPCs: ${d.data.npc_count}<br>`;
            content += `Connections: ${d.data.connections.length}<br>`;
            
            // Show agents currently at this location
            if (this.currentData && this.currentData.simulacra) {
                const agentsHere = Object.values(this.currentData.simulacra)
                    .filter(agent => agent.current_location === d.id);
                if (agentsHere.length > 0) {
                    content += `<strong>Agents here:</strong> ${agentsHere.map(a => a.name).join(', ')}<br>`;
                }
            }
        }
        
        this.tooltip
            .style("display", "block")
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px")
            .html(content);
    }
    
    hideTooltip() {
        this.tooltip.style("display", "none");
    }
    
    handleNodeClick(event, d) {
        if (d.type === 'agent') {
            this.requestAgentDetails(d.id);
        } else if (d.type === 'location') {
            this.requestLocationDetails(d.id);
        }
    }
    
    requestAgentDetails(agentId) {
        if (this.websocket && this.isConnected) {
            this.websocket.send(JSON.stringify({
                type: 'get_agent_details',
                agent_id: agentId
            }));
        }
    }
    
    requestLocationDetails(locationId) {
        if (this.websocket && this.isConnected) {
            this.websocket.send(JSON.stringify({
                type: 'get_location_details',
                location_id: locationId
            }));
        }
    }
    
    updateAgentsList(simulacra) {
        const container = document.getElementById('agentsList');
        container.innerHTML = '';
        
        Object.values(simulacra).forEach(agent => {
            const item = document.createElement('div');
            item.className = 'agent-item';
            item.onclick = () => this.requestAgentDetails(agent.id);
            
            item.innerHTML = `
                <div class="item-title">${agent.name}</div>
                <div class="item-details">
                    Status: ${agent.status}<br>
                    Location: ${agent.current_location}<br>
                    ${agent.current_action ? `Action: ${agent.current_action.substring(0, 60)}...` : ''}
                </div>
            `;
            
            container.appendChild(item);
        });
    }
    
    updateLocationsList(locations) {
        const container = document.getElementById('locationsList');
        container.innerHTML = '';
        
        Object.values(locations).forEach(location => {
            const item = document.createElement('div');
            item.className = 'location-item';
            item.onclick = () => this.requestLocationDetails(location.id);
            
            item.innerHTML = `
                <div class="item-title">${location.name}</div>
                <div class="item-details">
                    Objects: ${location.object_count}<br>
                    NPCs: ${location.npc_count}<br>
                    Connections: ${location.connections.length}
                </div>
            `;
            
            container.appendChild(item);
        });
    }
    
    updateNarrativeList(narrative) {
        const container = document.getElementById('narrativeList');
        container.innerHTML = '';
        
        narrative.slice(-5).forEach(entry => {
            const item = document.createElement('div');
            item.className = 'narrative-item';
            item.textContent = entry.length > 200 ? entry.substring(0, 200) + '...' : entry;
            container.appendChild(item);
        });
    }
    
    updateWorldStatus(data) {
        const container = document.getElementById('worldStatus');
        container.innerHTML = `
            <div><strong>Active Agents:</strong> ${data.active_simulacra_count}</div>
            <div><strong>Locations:</strong> ${data.total_locations}</div>
            <div><strong>Objects:</strong> ${data.total_objects}</div>
            <div><strong>World Time:</strong> ${Math.round(data.world_time)}s</div>
        `;
        
        if (data.world_feeds && data.world_feeds.weather) {
            container.innerHTML += `
                <div><strong>Weather:</strong> ${data.world_feeds.weather.condition} (${data.world_feeds.weather.temperature_celsius}°C)</div>
            `;
        }
    }
    
    updateClock() {
        const now = new Date();
        document.getElementById('realTime').textContent = now.toLocaleTimeString();
    }
    
    handleResize() {
        this.width = window.innerWidth - 350;
        this.height = window.innerHeight - 50;
        
        this.svg
            .attr("width", this.width)
            .attr("height", this.height);
        
        this.simulation
            .force("center", d3.forceCenter(this.width / 2, this.height / 2))
            .force("x", d3.forceX(this.width / 2).strength(0.1))
            .force("y", d3.forceY(this.height / 2).strength(0.1))
            .alpha(0.3)
            .restart();
    }
    
    // Drag functions
    dragStarted(event, d) {
        if (!event.active) this.simulation.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
    }
    
    dragged(event, d) {
        d.fx = event.x;
        d.fy = event.y;
    }
    
    dragEnded(event, d) {
        if (!event.active) this.simulation.alphaTarget(0);
        d.fx = null;
        d.fy = null;
    }
}

// Global control functions
function resetView() {
    visualizer.svg.transition().duration(750).call(
        visualizer.zoom.transform,
        d3.zoomIdentity
    );
}

function togglePause() {
    visualizer.isPaused = !visualizer.isPaused;
    console.log(visualizer.isPaused ? 'Visualization paused' : 'Visualization resumed');
}

function centerOnAgents() {
    if (!visualizer.currentData || !visualizer.currentData.simulacra) return;
    
    const agents = Object.values(visualizer.currentData.simulacra);
    if (agents.length === 0) return;
    
    // Find center of all agents
    const agentNodes = visualizer.nodesGroup.selectAll('.node-agent').nodes();
    if (agentNodes.length === 0) return;
    
    let avgX = 0, avgY = 0;
    agentNodes.forEach(node => {
        const d = d3.select(node).datum();
        avgX += d.x;
        avgY += d.y;
    });
    avgX /= agentNodes.length;
    avgY /= agentNodes.length;
    
    // Center view on agents
    const scale = 1.5;
    const translateX = visualizer.width / 2 - avgX * scale;
    const translateY = visualizer.height / 2 - avgY * scale;
    
    visualizer.svg.transition().duration(750).call(
        visualizer.zoom.transform,
        d3.zoomIdentity.translate(translateX, translateY).scale(scale)
    );
}

// Initialize visualizer when page loads
let visualizer;
document.addEventListener('DOMContentLoaded', () => {
    visualizer = new SimulationVisualizer();
});