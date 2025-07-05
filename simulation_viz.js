// TheSimulation Web Visualizer
// Real-time D3.js visualization with WebSocket connection

class SimulationVisualizer {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.isPaused = false;
        this.currentData = null;
        this.commandHistory = [];
        this.selectedAgent = null;
        this.interactiveMode = false;
        this.chatMessages = [];
        this.lastCheckTime = 0;
        this.lastNarrativeCount = 0;
        this.commandResponses = [];
        
        // D3 setup
        this.svg = d3.select("#visualization");
        this.width = window.innerWidth - 420; // Subtract info panel width
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
                const data = JSON.parse(event.data);
                
                // Debug logging
                if (data.type === 'command_response' || data.type === 'chat_response') {
                    console.log('WebSocket message received:', data.type, data);
                }
                
                // Handle different message types
                if (data.type === 'command_response') {
                    this.handleCommandResponse(data);
                } else if (data.type === 'chat_response') {
                    this.handleChatResponse(data);
                } else if (!this.isPaused) {
                    // Regular simulation update
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
        this.updateLocationsList(data.locations || {});
        this.updateNarrativeList(data.recent_narrative);
        this.updateWorldStatus(data);
        this.updateFeedPanels(data);
    }
    
    updateVisualization(data) {
        // Prepare nodes and links
        const nodes = [];
        const links = [];
        
        // Extract locations from the correct state structure
        const locations = data.locations || {};
        
        // Add location nodes
        Object.values(locations).forEach(location => {
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
            if (agent.current_location && locations[agent.current_location]) {
                const location = locations[agent.current_location];
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
            if (agent.current_location && locations[agent.current_location]) {
                links.push({
                    source: agent.id,
                    target: agent.current_location,
                    type: 'agent-location'
                });
            }
        });
        
        // Add location connections
        Object.values(locations).forEach(location => {
            // Check for connections field (as provided by WebSocket server)
            const connections = location.connections || [];
            if (Array.isArray(connections)) {
                connections.forEach(connection => {
                    // Handle both string IDs and connection objects
                    const targetId = typeof connection === 'string' 
                        ? connection 
                        : connection.to_location_id_hint || connection.target_id || connection.id;
                    
                    if (targetId && locations[targetId]) {
                        // Avoid duplicate bidirectional links
                        const linkExists = links.some(l => {
                            const sourceId = l.source.id || l.source;
                            const targetId_l = l.target.id || l.target;
                            return (sourceId === location.id && targetId_l === targetId) ||
                                   (sourceId === targetId && targetId_l === location.id);
                        });

                        if (!linkExists) {
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
            content += `Connections: ${(d.data.connections || []).length}<br>`;
            
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
                    Objects: ${location.object_count || 0}<br>
                    NPCs: ${location.npc_count || 0}<br>
                    Connections: ${(location.connections || []).length}
                </div>
            `;
            
            container.appendChild(item);
        });
    }
    
    updateNarrativeList(narrative) {
        const container = document.getElementById('narrativeList');
        
        // Check if we have new narrative entries
        const hasNewEntries = narrative.length > this.lastNarrativeCount;
        const newEntriesCount = hasNewEntries ? narrative.length - this.lastNarrativeCount : 0;
        
        // Clear existing content
        container.innerHTML = '';
        
        // Show recent narrative entries (last 8 entries)
        const recentEntries = narrative.slice(-8);
        
        recentEntries.forEach((entry, index) => {
            const item = document.createElement('div');
            
            // Mark new entries with special styling
            const isNewEntry = hasNewEntries && index >= recentEntries.length - newEntriesCount;
            item.className = `narrative-item ${isNewEntry ? 'new' : ''}`;
            
            // Parse and format the narrative entry
            const formattedEntry = this.formatNarrativeEntry(entry);
            item.innerHTML = formattedEntry;
            
            container.appendChild(item);
        });
        
        // Update our tracking
        this.lastNarrativeCount = narrative.length;
        
        // Auto-scroll to bottom to show newest entries
        container.scrollTop = container.scrollHeight;
    }

    formatNarrativeEntry(entry) {
        // Parse timestamp and content from entries like "[T123.4] Some content"
        const timeMatch = entry.match(/^\[T([\d.]+)\]\s*(.+)$/);
        if (timeMatch) {
            const simTime = parseFloat(timeMatch[1]);
            const content = timeMatch[2];
            
            // Format different types of content
            let formattedContent = content;
            
            // Highlight agent names and actions
            formattedContent = formattedContent.replace(
                /(sim_\w+|Agent \w+)/g, 
                '<strong style="color: #ff6b6b;">$1</strong>'
            );
            
            // Highlight action verbs
            formattedContent = formattedContent.replace(
                /\b(decides to|says|thinks|moves to|enters|exits|picks up|drops|looks at)\b/gi,
                '<em style="color: #007bff;">$1</em>'
            );
            
            // Highlight locations
            formattedContent = formattedContent.replace(
                /\b([A-Z][a-zA-Z]*_\d+|Home|Office|Park|Kitchen|Bedroom)\b/g,
                '<span style="color: #69b3a2; font-weight: 500;">$1</span>'
            );
            
            // Special formatting for different event types
            if (content.includes('[InteractionMode]')) {
                formattedContent = formattedContent.replace('[InteractionMode]', 
                    '<span style="background: #fff3cd; padding: 2px 4px; border-radius: 3px; font-size: 10px;">[INTERACTIVE]</span>');
            }
            
            if (content.includes('[System Teleport]')) {
                formattedContent = formattedContent.replace('[System Teleport]', 
                    '<span style="background: #d4edda; padding: 2px 4px; border-radius: 3px; font-size: 10px;">[TELEPORT]</span>');
            }
            
            return `
                <div style="font-size: 10px; color: #666; margin-bottom: 3px;">T${simTime.toFixed(1)}s</div>
                <div>${formattedContent}</div>
            `;
        } else {
            // Fallback for entries without timestamp
            return entry.length > 200 ? entry.substring(0, 200) + '...' : entry;
        }
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
        this.width = window.innerWidth - 420;
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

    // Command handling methods
    sendCommand(command) {
        if (this.websocket && this.isConnected) {
            // Add timestamp to command
            command.timestamp = Date.now();
            console.log('Sending command:', command);
            this.websocket.send(JSON.stringify(command));
            
            // Add to history
            this.commandHistory.push({
                command: command,
                timestamp: new Date().toLocaleTimeString(),
                status: 'pending'
            });
            
            this.showNotification('Command sent...', false);
        } else {
            alert('Not connected to simulation server');
        }
    }

    handleCommandResponse(data) {
        console.log('Received command response:', data);
        
        // Find the corresponding command in history and update its status
        const historyItem = this.commandHistory.find(item => 
            item.status === 'pending'
        );
        
        if (historyItem) {
            historyItem.response = data;
            historyItem.status = data.success ? 'success' : 'error';
        }

        // Add to command responses display
        this.addCommandResponse(data, historyItem?.command);

        // Show response notification with more details
        let message = data.success ? 
            'Command executed successfully' : 
            `Command failed: ${data.message || 'Unknown error'}`;
        
        // Add specific success details for different command types
        if (data.success && historyItem?.command) {
            const commandType = historyItem.command.command;
            switch (commandType) {
                case 'narrate':
                    message = 'Narrative injected - agents will respond to the new event';
                    break;
                case 'inject_event':
                    message = `Event sent to ${historyItem.command.agent_id} - check narrative feed for response`;
                    break;
                case 'teleport_agent':
                    message = `${historyItem.command.agent_id} teleported to ${historyItem.command.new_location_id}`;
                    break;
                case 'world_info':
                    message = `World ${historyItem.command.category} updated - agents may notice changes`;
                    break;
            }
        }
        
        this.showNotification(message, !data.success);
        
        // Log response for debugging
        if (data.success) {
            console.log('Command successful:', data.message);
        } else {
            console.error('Command failed:', data.message);
        }
    }

    handleChatResponse(data) {
        if (this.interactiveMode && data.agent_id === this.selectedAgent) {
            this.addChatMessage('agent', data.content);
        }
    }

    showNotification(message, isError = false) {
        // Create notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${isError ? '#dc3545' : '#28a745'};
            color: white;
            padding: 12px 20px;
            border-radius: 6px;
            z-index: 3000;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
            font-size: 14px;
            font-weight: 500;
            max-width: 300px;
            word-wrap: break-word;
            opacity: 0;
            transform: translateX(100%);
            transition: all 0.3s ease;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        // Trigger animation
        setTimeout(() => {
            notification.style.opacity = '1';
            notification.style.transform = 'translateX(0)';
        }, 10);
        
        // Remove after delay
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 300);
        }, 3000);
    }

    populateAgentList(containerId) {
        const container = document.getElementById(containerId);
        container.innerHTML = '';
        
        if (!this.currentData || !this.currentData.simulacra) {
            container.innerHTML = '<div style="padding: 10px; color: #666;">No agents available</div>';
            return;
        }

        Object.values(this.currentData.simulacra).forEach(agent => {
            const option = document.createElement('div');
            option.className = 'agent-option';
            option.innerHTML = `
                <div style="font-weight: bold;">${agent.name}</div>
                <div style="font-size: 11px; color: #666;">Status: ${agent.status} | Location: ${agent.current_location}</div>
            `;
            option.onclick = () => {
                // Remove selection from other options
                container.querySelectorAll('.agent-option').forEach(opt => opt.classList.remove('selected'));
                option.classList.add('selected');
                this.selectedAgent = agent.id;
            };
            container.appendChild(option);
        });
    }

    addChatMessage(type, content) {
        const messagesContainer = document.getElementById('chatMessages');
        const message = document.createElement('div');
        message.className = `chat-message ${type}`;
        message.textContent = content;
        messagesContainer.appendChild(message);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
    }

    addCommandResponse(responseData, command) {
        const container = document.getElementById('responseList');
        const responseItem = document.createElement('div');
        responseItem.className = `response-item ${responseData.success ? 'success' : 'error'}`;
        
        const commandName = command ? command.command : 'Unknown';
        const timestamp = new Date().toLocaleTimeString();
        
        let responseContent = responseData.message || 'No message';
        let detailsHtml = '';
        
        // Add specific content based on command type
        if (responseData.success && command) {
            switch (command.command) {
                case 'narrate':
                    responseContent = `Narrative "${command.text.substring(0, 50)}..." was injected into simulation`;
                    break;
                case 'inject_event':
                    responseContent = `Event "${command.description.substring(0, 50)}..." sent to ${command.agent_id}`;
                    break;
                case 'teleport_agent':
                    responseContent = `${command.agent_id} teleported to ${command.new_location_id}`;
                    if (command.location_description) {
                        detailsHtml = `<div class="response-details">Location: ${command.location_description}</div>`;
                    }
                    break;
                case 'world_info':
                    responseContent = `${command.category} updated: "${command.info.substring(0, 100)}..."`;
                    break;
            }
        }
        
        // Show response data if available
        if (responseData.data && Object.keys(responseData.data).length > 0) {
            detailsHtml += `<div class="response-details">Response data: ${JSON.stringify(responseData.data, null, 2)}</div>`;
        }
        
        responseItem.innerHTML = `
            <div class="response-time">${timestamp}</div>
            <div class="response-header">${commandName.toUpperCase()}</div>
            <div class="response-content">${responseContent}</div>
            ${detailsHtml}
        `;
        
        container.insertBefore(responseItem, container.firstChild);
        
        // Keep only last 10 responses
        while (container.children.length > 10) {
            container.removeChild(container.lastChild);
        }
        
        // Add to internal array
        this.commandResponses.unshift({
            timestamp: timestamp,
            command: command,
            response: responseData
        });
        
        if (this.commandResponses.length > 20) {
            this.commandResponses = this.commandResponses.slice(0, 20);
        }
    }

    updateFeedPanels(data) {
        // Update Latest Simulacra
        const simulacraElement = document.getElementById('latestSimulacra');
        if (data.simulacra && Object.keys(data.simulacra).length > 0) {
            const agents = Object.values(data.simulacra);
            // Find most recently active agent or first one
            const latestAgent = agents.find(agent => agent.status === 'busy') || agents[0];
            const simulacraText = `${latestAgent.name}: ${latestAgent.status}\n${latestAgent.current_action || 'No current action'}`;
            if (simulacraElement.textContent !== simulacraText) {
                simulacraElement.textContent = simulacraText;
                simulacraElement.classList.add('updated');
                setTimeout(() => simulacraElement.classList.remove('updated'), 500);
            }
        } else {
            simulacraElement.textContent = 'No agents active';
        }

        // Update World Engine
        const worldEngineElement = document.getElementById('latestWorldEngine');
        const worldEngineText = data.world_engine_activity || 'No recent activity';
        if (worldEngineElement.textContent !== worldEngineText) {
            worldEngineElement.textContent = worldEngineText;
            worldEngineElement.classList.add('updated');
            setTimeout(() => worldEngineElement.classList.remove('updated'), 500);
        }

        // Update Narrative
        const narrativeElement = document.getElementById('latestNarrative');
        if (data.recent_narrative && data.recent_narrative.length > 0) {
            const latestNarrative = data.recent_narrative[data.recent_narrative.length - 1];
            const narrativeText = typeof latestNarrative === 'string' ? latestNarrative : latestNarrative.text || 'No narrative';
            if (narrativeElement.textContent !== narrativeText) {
                narrativeElement.textContent = narrativeText;
                narrativeElement.classList.add('updated');
                setTimeout(() => narrativeElement.classList.remove('updated'), 500);
            }
        } else {
            narrativeElement.textContent = 'No recent narrative';
        }
    }
}

// Modal and command functions
function showModal(modalId) {
    document.getElementById(modalId).style.display = 'block';
}

function closeModal(modalId) {
    document.getElementById(modalId).style.display = 'none';
}

function showNarrativeModal() {
    showModal('narrativeModal');
    document.getElementById('narrativeText').focus();
}

function sendNarrative() {
    const text = document.getElementById('narrativeText').value;
    if (!text.trim()) {
        alert('Please enter narrative text');
        return;
    }
    
    visualizer.sendCommand({
        command: 'narrate',
        text: text
    });
    
    closeModal('narrativeModal');
    document.getElementById('narrativeText').value = '';
}

function showAgentEventModal() {
    visualizer.populateAgentList('agentEventList');
    showModal('agentEventModal');
}

function sendAgentEvent() {
    if (!visualizer.selectedAgent) {
        alert('Please select an agent');
        return;
    }
    
    const description = document.getElementById('eventDescription').value;
    if (!description.trim()) {
        alert('Please enter event description');
        return;
    }
    
    visualizer.sendCommand({
        command: 'inject_event',
        agent_id: visualizer.selectedAgent,
        description: description
    });
    
    closeModal('agentEventModal');
    document.getElementById('eventDescription').value = '';
    visualizer.selectedAgent = null;
}

function showWorldInfoModal() {
    showModal('worldInfoModal');
    document.getElementById('worldInfoText').focus();
}

function sendWorldInfo() {
    const category = document.getElementById('worldInfoCategory').value;
    const info = document.getElementById('worldInfoText').value;
    
    if (!info.trim()) {
        alert('Please enter world information');
        return;
    }
    
    visualizer.sendCommand({
        command: 'world_info',
        category: category,
        info: info
    });
    
    closeModal('worldInfoModal');
    document.getElementById('worldInfoText').value = '';
}

function showTeleportModal() {
    visualizer.populateAgentList('teleportAgentList');
    showModal('teleportModal');
}

function sendTeleport() {
    if (!visualizer.selectedAgent) {
        alert('Please select an agent');
        return;
    }
    
    const location = document.getElementById('teleportLocation').value;
    if (!location.trim()) {
        alert('Please enter a location ID');
        return;
    }
    
    const description = document.getElementById('teleportDescription').value;
    
    visualizer.sendCommand({
        command: 'teleport_agent',
        agent_id: visualizer.selectedAgent,
        new_location_id: location,
        location_description: description || 'No specific description provided'
    });
    
    closeModal('teleportModal');
    document.getElementById('teleportLocation').value = '';
    document.getElementById('teleportDescription').value = '';
    visualizer.selectedAgent = null;
}

function showInteractiveChatModal() {
    visualizer.populateAgentList('chatAgentList');
    showModal('interactiveChatModal');
    document.getElementById('chatMessages').innerHTML = '';
    visualizer.chatMessages = [];
}

function startInteractiveChat() {
    if (!visualizer.selectedAgent) {
        alert('Please select an agent first');
        return;
    }
    
    visualizer.interactiveMode = true;
    visualizer.sendCommand({
        command: 'start_interaction_mode',
        agent_id: visualizer.selectedAgent
    });
    
    visualizer.addChatMessage('system', `Started interactive chat with ${visualizer.selectedAgent}`);
    document.getElementById('chatInput').focus();
}

function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;
    
    if (!visualizer.interactiveMode) {
        startInteractiveChat();
        return;
    }
    
    // Parse message type and content
    let eventType = 'text_message';
    let content = message;
    
    if (message.startsWith('text:')) {
        eventType = 'text_message';
        content = message.substring(5).trim();
    } else if (message.startsWith('call:')) {
        eventType = 'phone_call';
        content = message.substring(5).trim();
    } else if (message.startsWith('voice:')) {
        eventType = 'voice';
        content = message.substring(6).trim();
    } else if (message.startsWith('event:')) {
        eventType = 'custom';
        content = message.substring(6).trim();
    } else if (message.startsWith('doorbell:')) {
        eventType = 'doorbell';
        content = message.substring(9).trim();
    } else if (message.startsWith('noise:')) {
        eventType = 'noise';
        content = message.substring(6).trim();
    }
    
    visualizer.sendCommand({
        command: 'interaction_event',
        agent_id: visualizer.selectedAgent,
        event_type: eventType,
        content: content
    });
    
    visualizer.addChatMessage('user', content);
    input.value = '';
}

function closeInteractiveChat() {
    if (visualizer.interactiveMode) {
        visualizer.sendCommand({
            command: 'end_interaction_mode',
            agent_id: visualizer.selectedAgent
        });
        visualizer.interactiveMode = false;
    }
    
    closeModal('interactiveChatModal');
    visualizer.selectedAgent = null;
}

function showHistoryModal() {
    updateHistoryDisplay();
    showModal('historyModal');
}

function updateHistoryDisplay() {
    const container = document.getElementById('historyList');
    container.innerHTML = '';
    
    if (visualizer.commandHistory.length === 0) {
        container.innerHTML = '<div style="padding: 20px; text-align: center; color: #666;">No commands executed yet</div>';
        return;
    }
    
    visualizer.commandHistory.slice().reverse().forEach((item, index) => {
        const historyItem = document.createElement('div');
        historyItem.className = `history-item ${item.status}`;
        historyItem.innerHTML = `
            <div style="font-weight: bold;">${item.command.command}</div>
            <div style="margin: 5px 0; font-size: 10px; color: #666;">
                ${item.timestamp} - Status: ${item.status}
            </div>
            ${item.response ? `<div style="font-size: 11px;">${JSON.stringify(item.response, null, 2)}</div>` : ''}
        `;
        container.appendChild(historyItem);
    });
}

function clearHistory() {
    if (confirm('Are you sure you want to clear the command history?')) {
        visualizer.commandHistory = [];
        updateHistoryDisplay();
    }
}


// Enhanced chat input handler
document.addEventListener('DOMContentLoaded', () => {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                sendChatMessage();
            }
        });
    }
});

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