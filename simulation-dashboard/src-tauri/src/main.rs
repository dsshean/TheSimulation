use tauri::{Emitter, State};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tokio::time::Duration;
use redis::AsyncCommands;
use futures::StreamExt;

#[derive(Debug, Serialize, Deserialize, Clone)]
struct SimulationState {
    world_time: f64,
    world_instance_uuid: String,
    simulacra_profiles: serde_json::Value,
    active_simulacra_ids: Vec<String>,
    current_world_state: serde_json::Value,
    world_feeds: serde_json::Value,
    world_template_details: serde_json::Value,
    narrative_log: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CommandResponse {
    success: bool,
    message: String,
    data: Option<serde_json::Value>,
}

struct SimulationManager {
    redis_client: Arc<Mutex<Option<redis::Client>>>,
    state: Arc<Mutex<Option<SimulationState>>>,
    running: Arc<Mutex<bool>>,
}

impl SimulationManager {
    fn new() -> Self {
        Self {
            redis_client: Arc::new(Mutex::new(None)),
            state: Arc::new(Mutex::new(None)),
            running: Arc::new(Mutex::new(false)),
        }
    }
    
    async fn connect_redis(&self) -> Result<(), redis::RedisError> {
        let client = redis::Client::open("redis://127.0.0.1:6379/")?;
        
        // Test connection
        let mut conn = client.get_async_connection().await?;
        let _: String = redis::cmd("PING").query_async(&mut conn).await?;
        
        {
            let mut redis_client = self.redis_client.lock().unwrap();
            *redis_client = Some(client);
        }
        
        Ok(())
    }
}

#[tauri::command]
async fn start_simulation(
    manager: State<'_, Arc<SimulationManager>>,
    app: tauri::AppHandle,
) -> Result<CommandResponse, String> {
    // Check if already running
    {
        let running = manager.running.lock().unwrap();
        if *running {
            return Ok(CommandResponse {
                success: false,
                message: "Redis connection already active".to_string(),
                data: None,
            });
        }
    }

    // Connect to Redis
    match manager.connect_redis().await {
        Ok(_) => {
            // Set running flag after successful connection
            {
                let mut running = manager.running.lock().unwrap();
                *running = true;
            }
            
            // Start Redis state monitoring task
            let manager_clone = manager.inner().clone();
            let app_clone = app.clone();
            tokio::spawn(async move {
                redis_state_monitor(manager_clone, app_clone).await;
            });

            Ok(CommandResponse {
                success: true,
                message: "Connected to simulation via Redis".to_string(),
                data: None,
            })
        }
        Err(e) => Ok(CommandResponse {
            success: false,
            message: format!("Failed to connect to Redis: {}", e),
            data: None,
        }),
    }
}

#[tauri::command]
async fn stop_simulation(manager: State<'_, Arc<SimulationManager>>) -> Result<CommandResponse, String> {
    let mut running = manager.running.lock().unwrap();
    
    if *running {
        *running = false;
        drop(running); // Explicitly drop the mutex guard
        // Redis connection will be closed in monitor task
        Ok(CommandResponse {
            success: true,
            message: "Disconnected from simulation".to_string(),
            data: None,
        })
    } else {
        Ok(CommandResponse {
            success: false,
            message: "Not connected to simulation".to_string(),
            data: None,
        })
    }
}

#[tauri::command]
async fn get_simulation_state(manager: State<'_, Arc<SimulationManager>>) -> Result<Option<SimulationState>, String> {
    let state = manager.state.lock().unwrap();
    Ok(state.clone())
}

#[tauri::command]
async fn inject_narrative(
    manager: State<'_, Arc<SimulationManager>>,
    text: String,
) -> Result<CommandResponse, String> {
    send_redis_command(&manager, "inject_narrative", serde_json::json!({"text": text})).await
}

#[tauri::command]
async fn send_agent_event(
    manager: State<'_, Arc<SimulationManager>>,
    agent_id: String,
    description: String,
) -> Result<CommandResponse, String> {
    send_redis_command(&manager, "send_agent_event", serde_json::json!({
        "agent_id": agent_id,
        "description": description
    })).await
}

#[tauri::command]
async fn update_world_info(
    manager: State<'_, Arc<SimulationManager>>,
    category: String,
    info: String,
) -> Result<CommandResponse, String> {
    send_redis_command(&manager, "update_world_info", serde_json::json!({
        "category": category,
        "info": info
    })).await
}

#[tauri::command]
async fn teleport_agent(
    manager: State<'_, Arc<SimulationManager>>,
    agent_id: String,
    location: String,
) -> Result<CommandResponse, String> {
    send_redis_command(&manager, "teleport_agent", serde_json::json!({
        "agent_id": agent_id,
        "location": location
    })).await
}

#[tauri::command]
async fn interactive_chat(
    manager: State<'_, Arc<SimulationManager>>,
    agent_id: String,
    message: String,
) -> Result<CommandResponse, String> {
    send_redis_command(&manager, "interactive_chat", serde_json::json!({
        "agent_id": agent_id,
        "message": message
    })).await
}

#[tauri::command]
async fn get_latest_narrative_image() -> Result<String, String> {
    get_narrative_image_by_index(0).await
}

#[tauri::command]
async fn get_narrative_image_by_index(index: usize) -> Result<String, String> {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::env;
    
    // Try multiple possible paths for the images directory
    let possible_paths = vec![
        PathBuf::from("data/narrative_images"),
        PathBuf::from("../data/narrative_images"),
        PathBuf::from("../../data/narrative_images"),
        PathBuf::from("/home/doug/TheSimulation/data/narrative_images"),
    ];
    
    let mut images_dir: Option<PathBuf> = None;
    
    for path in possible_paths {
        if path.exists() && path.is_dir() {
            images_dir = Some(path);
            break;
        }
    }
    
    let images_dir = images_dir.ok_or_else(|| {
        let current_dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("unknown"));
        format!("Images directory not found. Current working directory: {:?}", current_dir)
    })?;
    
    // Get all image files sorted by modification time (newest first)
    let mut image_files: Vec<(std::time::SystemTime, std::path::PathBuf)> = Vec::new();
    
    let entries = fs::read_dir(&images_dir)
        .map_err(|e| format!("Failed to read images directory: {}", e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
        let path = entry.path();
        
        if path.is_file() && path.extension().map_or(false, |ext| ext == "png") {
            let metadata = fs::metadata(&path)
                .map_err(|e| format!("Failed to read file metadata: {}", e))?;
            
            if let Ok(modified) = metadata.modified() {
                image_files.push((modified, path));
            }
        }
    }
    
    // Sort by modification time (newest first)
    image_files.sort_by(|a, b| b.0.cmp(&a.0));
    
    if image_files.is_empty() {
        return Err("No image files found in directory".to_string());
    }
    
    if index >= image_files.len() {
        return Err(format!("Image index {} out of range (0-{})", index, image_files.len() - 1));
    }
    
    let (_, image_path) = &image_files[index];
    
    // Read the image file and convert to base64
    let image_data = fs::read(image_path)
        .map_err(|e| format!("Failed to read image file: {}", e))?;
    
    let base64_data = base64::encode(&image_data);
    let data_url = format!("data:image/png;base64,{}", base64_data);
    
    Ok(data_url)
}

#[tauri::command]
async fn get_narrative_images_count() -> Result<usize, String> {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::env;
    
    // Try multiple possible paths for the images directory
    let possible_paths = vec![
        PathBuf::from("data/narrative_images"),
        PathBuf::from("../data/narrative_images"),
        PathBuf::from("../../data/narrative_images"),
        PathBuf::from("/home/doug/TheSimulation/data/narrative_images"),
    ];
    
    let mut images_dir: Option<PathBuf> = None;
    
    for path in possible_paths {
        if path.exists() && path.is_dir() {
            images_dir = Some(path);
            break;
        }
    }
    
    let images_dir = images_dir.ok_or_else(|| {
        let current_dir = env::current_dir().unwrap_or_else(|_| PathBuf::from("unknown"));
        format!("Images directory not found. Current working directory: {:?}", current_dir)
    })?;
    
    // Count PNG files
    let mut count = 0;
    let entries = fs::read_dir(&images_dir)
        .map_err(|e| format!("Failed to read images directory: {}", e))?;
    
    for entry in entries {
        let entry = entry.map_err(|e| format!("Failed to read directory entry: {}", e))?;
        let path = entry.path();
        
        if path.is_file() && path.extension().map_or(false, |ext| ext == "png") {
            count += 1;
        }
    }
    
    Ok(count)
}

async fn send_redis_command(
    manager: &State<'_, Arc<SimulationManager>>,
    command: &str,
    params: serde_json::Value,
) -> Result<CommandResponse, String> {
    let redis_client = {
        let client_lock = manager.redis_client.lock().unwrap();
        client_lock.clone()
    };
    
    if let Some(client) = redis_client {
        match client.get_async_connection().await {
            Ok(mut conn) => {
                let request_id = uuid::Uuid::new_v4().to_string();
                let mut command_data = serde_json::json!({
                    "command": command,
                    "request_id": request_id
                });
                
                // Merge params into command_data
                if let Some(params_obj) = params.as_object() {
                    if let Some(command_obj) = command_data.as_object_mut() {
                        for (key, value) in params_obj {
                            command_obj.insert(key.clone(), value.clone());
                        }
                    }
                }
                
                // Send command to Redis
                let command_json = serde_json::to_string(&command_data).unwrap();
                let _: () = conn.publish("simulation:commands", command_json).await
                    .map_err(|e| format!("Failed to send command: {}", e))?;
                
                // Wait for response (simplified - in production you'd want timeout)
                let response_channel = format!("simulation:response:{}", request_id);
                let mut pubsub = conn.into_pubsub();
                pubsub.subscribe(&response_channel).await
                    .map_err(|e| format!("Failed to subscribe to response: {}", e))?;
                
                // Wait for response with timeout
                let mut message_stream = pubsub.on_message();
                tokio::select! {
                    msg = message_stream.next() => {
                        if let Some(msg) = msg {
                            let data: String = msg.get_payload().unwrap_or_default();
                            if let Ok(response_data) = serde_json::from_str::<serde_json::Value>(&data) {
                                if let Some(response) = response_data.get("response") {
                                    return Ok(serde_json::from_value(response.clone()).unwrap_or(CommandResponse {
                                        success: false,
                                        message: "Invalid response format".to_string(),
                                        data: None,
                                    }));
                                }
                            }
                        }
                        Err("No valid response received".to_string())
                    }
                    _ = tokio::time::sleep(Duration::from_secs(10)) => {
                        Err("Command timeout".to_string())
                    }
                }
            }
            Err(e) => Err(format!("Redis connection error: {}", e)),
        }
    } else {
        Err("Not connected to Redis".to_string())
    }
}

async fn redis_state_monitor(manager: Arc<SimulationManager>, app: tauri::AppHandle) {
    let redis_client = {
        let client_lock = manager.redis_client.lock().unwrap();
        client_lock.clone()
    };
    
    if let Some(client) = redis_client {
        match client.get_async_connection().await {
            Ok(conn) => {
                let mut pubsub = conn.into_pubsub();
                if let Err(e) = pubsub.subscribe("simulation:state").await {
                    log::error!("Failed to subscribe to simulation:state: {}", e);
                    return;
                }
                
                log::info!("Redis state monitor started");
                
                loop {
                    let running = {
                        let r = manager.running.lock().unwrap();
                        *r
                    };
                    
                    if !running {
                        break;
                    }
                    
                    let mut message_stream = pubsub.on_message();
                    tokio::select! {
                        msg = message_stream.next() => {
                            if let Some(msg) = msg {
                                let data: String = msg.get_payload().unwrap_or_default();
                                if let Ok(state_data) = serde_json::from_str::<SimulationState>(&data) {
                                    {
                                        let mut state = manager.state.lock().unwrap();
                                        *state = Some(state_data.clone());
                                    }
                                    
                                    // Emit state update to frontend
                                    let _ = app.emit("simulation-state-update", &state_data);
                                }
                            }
                        }
                        _ = tokio::time::sleep(Duration::from_secs(1)) => {
                            // Keep loop alive and check running status
                        }
                    }
                }
                
                log::info!("Redis state monitor stopped");
            }
            Err(e) => {
                log::error!("Failed to create Redis pubsub connection: {}", e);
            }
        }
    }
}


#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::init();
    
    let simulation_manager = Arc::new(SimulationManager::new());
    
    tauri::Builder::default()
        .manage(simulation_manager)
        .invoke_handler(tauri::generate_handler![
            start_simulation,
            stop_simulation,
            get_simulation_state,
            inject_narrative,
            send_agent_event,
            update_world_info,
            teleport_agent,
            interactive_chat,
            get_latest_narrative_image,
            get_narrative_image_by_index,
            get_narrative_images_count
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

fn main() {
    run();
}