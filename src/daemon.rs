use anyhow::{Context, Result};
use nix::unistd::{getgid, getuid, Gid, Uid};
use serde::{Deserialize, Serialize};
use std::env;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;
use tracing::{debug, error, info, warn};

use crate::audio_input::AudioInput;
use crate::stt_client::{AudioBuffer, SttClient};
use crate::virtual_keyboard::{RealKeyboardHardware, VirtualKeyboard};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonCommand {
    StartListening,
    StopListening,
    Status,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonResponse {
    Ok,
    Error(String),
    Status { listening: bool },
}

pub struct Daemon {
    socket_path: PathBuf,
    keyboard: Arc<Mutex<Option<VirtualKeyboard<RealKeyboardHardware>>>>,
    listening: Arc<Mutex<bool>>,
    original_user: OriginalUser,
    voice_enter_enabled: bool,
    uppercase_enabled: bool,
}

impl Daemon {
    pub fn new(socket_path: PathBuf, original_user: OriginalUser) -> Result<Self> {
        // Create virtual keyboard while we have root privileges
        let hardware = RealKeyboardHardware::new("Voice Keyboard")
            .context("Failed to create keyboard hardware")?;
        let mut keyboard = VirtualKeyboard::new(hardware);
        // Default settings - can be made configurable later
        keyboard.set_voice_enter_enabled(true);
        keyboard.set_uppercase_enabled(false);

        Ok(Self {
            socket_path,
            keyboard: Arc::new(Mutex::new(Some(keyboard))),
            listening: Arc::new(Mutex::new(false)),
            original_user,
            voice_enter_enabled: true,
            uppercase_enabled: false,
        })
    }

    pub async fn run(&self) -> Result<()> {
        // Drop privileges before starting the daemon
        self.original_user
            .drop_privileges()
            .context("Failed to drop root privileges")?;

        // Remove old socket if it exists
        if self.socket_path.exists() {
            std::fs::remove_file(&self.socket_path)
                .context("Failed to remove old socket file")?;
        }

        // Create parent directory if needed
        if let Some(parent) = self.socket_path.parent() {
            std::fs::create_dir_all(parent)
                .context("Failed to create socket directory")?;
        }

        let listener = UnixListener::bind(&self.socket_path)
            .context("Failed to bind Unix socket")?;

        info!("Daemon listening on socket: {:?}", self.socket_path);

        // Set socket permissions so user can connect
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self.socket_path, std::fs::Permissions::from_mode(0o666))
                .context("Failed to set socket permissions")?;
        }

        loop {
            match listener.accept().await {
                Ok((stream, _addr)) => {
                    debug!("New client connected");
                    let keyboard = self.keyboard.clone();
                    let listening = self.listening.clone();
                    let socket_path = self.socket_path.clone();

                    let voice_enter = self.voice_enter_enabled;
                    let uppercase = self.uppercase_enabled;

                    tokio::spawn(async move {
                        if let Err(e) = Self::handle_client(stream, keyboard, listening, socket_path, voice_enter, uppercase).await {
                            error!("Error handling client: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Failed to accept connection: {}", e);
                }
            }
        }
    }

    async fn handle_client(
        mut stream: UnixStream,
        keyboard: Arc<Mutex<Option<VirtualKeyboard<RealKeyboardHardware>>>>,
        listening: Arc<Mutex<bool>>,
        socket_path: PathBuf,
        voice_enter_enabled: bool,
        uppercase_enabled: bool,
    ) -> Result<()> {
        let mut buf = vec![0u8; 4096];
        let n = stream.read(&mut buf).await?;
        if n == 0 {
            return Ok(());
        }

        let command: DaemonCommand = serde_json::from_slice(&buf[..n])
            .context("Failed to parse command")?;

        debug!("Received command: {:?}", command);

        let response = match command {
            DaemonCommand::StartListening => {
                let mut is_listening = listening.lock().await;
                if *is_listening {
                    DaemonResponse::Error("Already listening".to_string())
                } else {
                    *is_listening = true;
                    drop(is_listening);

                    // Play start sound
                    play_sound("start").await;

                    // Start listening in background
                    let keyboard_clone = keyboard.clone();
                    let listening_clone = listening.clone();
                    let socket_path_clone = socket_path.clone();

                    tokio::spawn(async move {
                        let result = Self::start_listening_task(
                            keyboard_clone,
                            listening_clone.clone(),
                            socket_path_clone,
                            voice_enter_enabled,
                            uppercase_enabled,
                        ).await;
                        
                        // Ensure listening flag is reset even on error
                        if let Err(e) = result {
                            error!("Error in listening task: {}", e);
                            let mut is_listening = listening_clone.lock().await;
                            *is_listening = false;
                        }
                    });

                    DaemonResponse::Ok
                }
            }
            DaemonCommand::StopListening => {
                let mut is_listening = listening.lock().await;
                *is_listening = false;
                DaemonResponse::Ok
            }
            DaemonCommand::Status => {
                let is_listening = listening.lock().await;
                DaemonResponse::Status {
                    listening: *is_listening,
                }
            }
        };

        let response_json = serde_json::to_vec(&response)?;
        stream.write_all(&response_json).await?;

        Ok(())
    }

    async fn start_listening_task(
        keyboard: Arc<Mutex<Option<VirtualKeyboard<RealKeyboardHardware>>>>,
        listening: Arc<Mutex<bool>>,
        _socket_path: PathBuf,
        _voice_enter_enabled: bool,
        _uppercase_enabled: bool,
    ) -> Result<()> {
        info!("Starting transcription session...");

        let stt_url = env::var("DEEPGRAM_STT_URL")
            .unwrap_or_else(|_| "wss://api.deepgram.com/v2/listen".to_string());

        // Create a channel for audio data from recording
        let (recording_tx, mut recording_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(32);
        let recording_tx_for_blocking = recording_tx.clone();

        // Log environment for debugging
        debug!("STT connection environment check:");
        debug!("  DEEPGRAM_API_KEY: {}", if env::var("DEEPGRAM_API_KEY").is_ok() { "set" } else { "not set" });
        debug!("  DEEPGRAM_STT_URL: {:?}", env::var("DEEPGRAM_STT_URL").ok());
        debug!("  XDG_RUNTIME_DIR: {:?}", env::var("XDG_RUNTIME_DIR").ok());
        debug!("  PULSE_RUNTIME_PATH: {:?}", env::var("PULSE_RUNTIME_PATH").ok());

        // Create audio input in a blocking task since it's not Send
        let sample_rate = tokio::task::spawn_blocking(move || -> Result<u32> {
            // Log environment for debugging
            debug!("Audio input environment check:");
            debug!("  XDG_RUNTIME_DIR: {:?}", env::var("XDG_RUNTIME_DIR").ok());
            debug!("  PULSE_RUNTIME_PATH: {:?}", env::var("PULSE_RUNTIME_PATH").ok());
            debug!("  HOME: {:?}", env::var("HOME").ok());
            debug!("  USER: {:?}", env::var("USER").ok());
            
            let mut audio_input = AudioInput::new()
                .context("Failed to create audio input. Make sure PipeWire/PulseAudio is running and accessible.")?;
            let rate = audio_input.get_sample_rate();
            let ch = audio_input.get_channels();
            
            let mut audio_buffer = AudioBuffer::new(rate, 160);
            
            audio_input.start_recording(move |data| {
                // Average stereo channels to mono
                let mono_data: Vec<f32> = if ch == 2 {
                    let mut mono = Vec::with_capacity(data.len() / 2);
                    for chunk in data.chunks_exact(2) {
                        mono.push((chunk[0] + chunk[1]) / 2.0);
                    }
                    mono
                } else {
                    data.to_vec()
                };

                // Create audio chunks and send them
                let chunks = audio_buffer.add_samples(&mono_data);
                for chunk in chunks {
                    if let Err(_e) = recording_tx_for_blocking.blocking_send(chunk) {
                        // Channel closed, stop sending
                    }
                }
            })?;
            
            // Keep audio_input alive
            std::mem::forget(audio_input);
            Ok(rate)
        }).await??;

        let stt_client = SttClient::new(&stt_url, sample_rate);

        // Track last transcription time and whether we've received any transcription
        // Use std::sync::Mutex since these are accessed from synchronous callbacks
        let last_transcription_time = Arc::new(std::sync::Mutex::new(std::time::Instant::now()));
        let has_received_transcription = Arc::new(std::sync::Mutex::new(false));
        let received_final_text = Arc::new(std::sync::Mutex::new(false));

        let keyboard_cb = keyboard.clone();
        let last_transcription_time_cb = last_transcription_time.clone();
        let has_received_transcription_cb = has_received_transcription.clone();
        let received_final_text_cb = received_final_text.clone();

        let connect_result = stt_client
            .connect_and_transcribe(move |result| {
                // Update transcription tracking
                {
                    let mut last_time = last_transcription_time_cb.lock().unwrap();
                    *last_time = std::time::Instant::now();
                }

                if !result.transcript.is_empty() {
                    let mut has_received = has_received_transcription_cb.lock().unwrap();
                    *has_received = true;
                }

                if result.event == "EndOfTurn" {
                    let mut received_final = received_final_text_cb.lock().unwrap();
                    *received_final = true;
                }

                // Handle transcription with keyboard
                if let Ok(mut kb_guard) = keyboard_cb.try_lock() {
                    if let Some(ref mut kb) = *kb_guard {
                        if result.event == "EndOfTurn" {
                            if let Err(e) = kb.finalize_transcript() {
                                error!("Failed to finalize transcript: {}", e);
                            }
                        } else {
                            if let Err(e) = kb.update_transcript(&result.transcript) {
                                error!("Failed to update transcript: {}", e);
                            }
                        }
                    }
                }
            })
            .await;

        let (audio_tx, handle) = connect_result
            .with_context(|| {
                let api_key_set = env::var("DEEPGRAM_API_KEY").is_ok();
                if !api_key_set {
                    "Failed to connect to STT service. DEEPGRAM_API_KEY environment variable is not set. Please set it before starting the daemon."
                } else {
                    "Failed to connect to STT service. Check your network connection and Deepgram API key."
                }
            })?;


        info!("Connected to Deepgram, listening for speech...");

        // Wrap audio_tx in Arc to allow controlled dropping
        let audio_tx_arc = Arc::new(Mutex::new(Some(audio_tx)));
        let audio_tx_arc_for_forward = audio_tx_arc.clone();
        let audio_tx_arc_for_timeout = audio_tx_arc.clone();

        // Forward audio from recording to STT
        tokio::spawn(async move {
            while let Some(chunk) = recording_rx.recv().await {
                let tx_guard = audio_tx_arc_for_forward.lock().await;
                if let Some(ref tx) = *tx_guard {
                    if let Err(_e) = tx.send(chunk).await {
                        debug!("Audio channel closed, stopping forwarding");
                        break; // STT connection closed
                    }
                } else {
                    // Audio tx was dropped, stop forwarding
                    break;
                }
            }
        });

        // Monitor for timeout: disconnect if no transcription or final text received
        let recording_tx_for_timeout = recording_tx.clone();
        let listening_timeout = listening.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

                let is_listening = listening_timeout.lock().await;
                if !*is_listening {
                    break;
                }
                drop(is_listening);

                let (should_disconnect, reason) = {
                    let last_time = last_transcription_time.lock().unwrap();
                    let has_received = has_received_transcription.lock().unwrap();
                    let received_final = received_final_text.lock().unwrap();

                    // If we've received final text, disconnect immediately
                    if *received_final {
                        (true, Some("Final text received"))
                    }
                    // If no transcription received within 3 seconds, disconnect
                    else if !*has_received && last_time.elapsed() > std::time::Duration::from_secs(3) {
                        (true, Some("No transcription received within 3 seconds"))
                    }
                    // If transcription received but no updates for 3 seconds, disconnect
                    else if *has_received && !*received_final && last_time.elapsed() > std::time::Duration::from_secs(3) {
                        (true, Some("No transcription updates for 3 seconds"))
                    } else {
                        (false, None)
                    }
                };

                if should_disconnect {
                    if let Some(reason) = reason {
                        warn!("{}, disconnecting...", reason);
                    }
                    // Close the audio channel to stop sending audio - this will trigger CloseStream
                    let mut tx_guard = audio_tx_arc_for_timeout.lock().await;
                    *tx_guard = None; // Drop the sender
                    drop(tx_guard);
                    
                    // Close the recording channel to stop audio capture
                    drop(recording_tx_for_timeout);
                    break;
                }
            }
        });

        // Keep the original audio_tx alive in this scope until we want to disconnect
        // The timeout task will drop it when needed
        let _audio_tx_keep_alive = audio_tx_arc.clone();

        // Wait for handle to complete or timeout
        let handle_result = tokio::time::timeout(
            std::time::Duration::from_secs(300), // 5 minute max
            handle,
        ).await;

        // Always mark as not listening, even if there was an error
        let mut is_listening = listening.lock().await;
        *is_listening = false;
        drop(is_listening);

        // Play disconnect sound
        play_sound("stop").await;

        match handle_result {
            Ok(Ok(Ok(()))) => {
                info!("Transcription session ended normally");
            }
            Ok(Ok(Err(e))) => {
                warn!("Transcription session ended with error: {}", e);
            }
            Ok(Err(e)) => {
                warn!("Transcription task join error: {}", e);
            }
            Err(_) => {
                warn!("Transcription session timed out");
            }
        }

        Ok(())
    }
}

pub async fn play_sound(kind: &str) {
    let sound_cmd = match kind {
        "start" => "paplay /usr/share/sounds/freedesktop/stereo/bell.oga 2>/dev/null || aplay /usr/share/sounds/alsa/Front_Left.wav 2>/dev/null || echo -e '\\a'",
        "stop" => "paplay /usr/share/sounds/freedesktop/stereo/complete.oga 2>/dev/null || aplay /usr/share/sounds/alsa/Front_Right.wav 2>/dev/null || echo -e '\\a'",
        _ => return,
    };

    let output = tokio::process::Command::new("sh")
        .arg("-c")
        .arg(sound_cmd)
        .output()
        .await;

    if let Err(e) = output {
        debug!("Failed to play sound: {}", e);
    }
}

#[derive(Debug, Clone)]
pub struct OriginalUser {
    pub uid: Uid,
    pub gid: Gid,
    pub home: Option<String>,
    pub user: Option<String>,
}

impl OriginalUser {
    pub fn capture() -> Self {
        let uid = if let Ok(sudo_uid) = env::var("SUDO_UID") {
            Uid::from_raw(sudo_uid.parse().unwrap_or_else(|_| getuid().as_raw()))
        } else {
            getuid()
        };

        let gid = if let Ok(sudo_gid) = env::var("SUDO_GID") {
            Gid::from_raw(sudo_gid.parse().unwrap_or_else(|_| getgid().as_raw()))
        } else {
            getgid()
        };

        let home = env::var("HOME").ok();
        let user = env::var("SUDO_USER").ok().or_else(|| env::var("USER").ok());

        Self { uid, gid, home, user }
    }

    pub fn drop_privileges(&self) -> Result<()> {
        use nix::unistd::{getuid, setgid, setuid};

        if getuid().is_root() {
            debug!(
                "Dropping root privileges to uid={}, gid={}",
                self.uid, self.gid
            );

            // Capture environment variables BEFORE dropping privileges
            let pulse_runtime_path = env::var("PULSE_RUNTIME_PATH").ok();
            let xdg_runtime_dir = env::var("XDG_RUNTIME_DIR").ok();
            let display = env::var("DISPLAY").ok();
            let wayland_display = env::var("WAYLAND_DISPLAY").ok();
            let dbus_session_bus_address = env::var("DBUS_SESSION_BUS_ADDRESS").ok();
            let deepgram_api_key = env::var("DEEPGRAM_API_KEY").ok();
            let deepgram_stt_url = env::var("DEEPGRAM_STT_URL").ok();

            // If XDG_RUNTIME_DIR wasn't set, try to construct it from the user's home
            let xdg_runtime_dir = xdg_runtime_dir.or_else(|| {
                // Try systemd user runtime directory first (most common)
                let systemd_runtime = format!("/run/user/{}", self.uid.as_raw());
                if std::path::Path::new(&systemd_runtime).exists() {
                    Some(systemd_runtime)
                } else if let Some(ref home) = self.home {
                    // Fallback to home directory runtime
                    let runtime_path = format!("{}/.local/share/runtime-dir", home);
                    if std::path::Path::new(&runtime_path).exists() {
                        Some(runtime_path)
                    } else {
                        None
                    }
                } else {
                    None
                }
            });

            setgid(self.gid).context("Failed to drop group privileges")?;
            setuid(self.uid).context("Failed to drop user privileges")?;

            // Set environment variables AFTER dropping privileges
            if let Some(ref home) = self.home {
                env::set_var("HOME", home);
            }
            if let Some(ref user) = self.user {
                env::set_var("USER", user);
            }

            if let Some(ref pulse_path) = pulse_runtime_path {
                env::set_var("PULSE_RUNTIME_PATH", pulse_path);
            }
            if let Some(ref xdg_path) = xdg_runtime_dir {
                env::set_var("XDG_RUNTIME_DIR", xdg_path);
                debug!("Set XDG_RUNTIME_DIR to: {}", xdg_path);
            }
            if let Some(ref disp) = display {
                env::set_var("DISPLAY", disp);
            }
            if let Some(ref wayland_disp) = wayland_display {
                env::set_var("WAYLAND_DISPLAY", wayland_disp);
            }
            if let Some(ref dbus_addr) = dbus_session_bus_address {
                env::set_var("DBUS_SESSION_BUS_ADDRESS", dbus_addr);
            }
            if let Some(ref api_key) = deepgram_api_key {
                env::set_var("DEEPGRAM_API_KEY", api_key);
                debug!("Preserved DEEPGRAM_API_KEY");
            }
            if let Some(ref stt_url) = deepgram_stt_url {
                env::set_var("DEEPGRAM_STT_URL", stt_url);
            }

            debug!("Successfully dropped privileges to user");
            debug!("Environment: HOME={:?}, XDG_RUNTIME_DIR={:?}, PULSE_RUNTIME_PATH={:?}, DEEPGRAM_API_KEY={}", 
                   self.home, xdg_runtime_dir, pulse_runtime_path,
                   if deepgram_api_key.is_some() { "set" } else { "not set" });
            std::thread::sleep(std::time::Duration::from_millis(100));
        } else {
            debug!("Not running as root, no privilege dropping needed");
        }

        Ok(())
    }
}

pub async fn send_command(socket_path: &PathBuf, command: DaemonCommand) -> Result<DaemonResponse> {
    let mut stream = UnixStream::connect(socket_path)
        .await
        .context("Failed to connect to daemon socket")?;

    let command_json = serde_json::to_vec(&command)?;
    stream.write_all(&command_json).await?;
    stream.shutdown().await?;

    let mut response_buf = Vec::new();
    stream.read_to_end(&mut response_buf).await?;

    let response: DaemonResponse = serde_json::from_slice(&response_buf)
        .context("Failed to parse daemon response")?;

    Ok(response)
}
