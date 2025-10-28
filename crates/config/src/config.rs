use std::{
    collections::BTreeMap,
    fs,
    ops::AddAssign,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, anyhow};
use deepseek_ocr_core::runtime::{DeviceKind, Precision};
use serde::{Deserialize, Serialize};

use crate::fs::{VirtualFileSystem, VirtualPath};

const DEFAULT_MODEL_ID: &str = "deepseek-ocr";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub models: ModelRegistry,
    pub inference: InferenceSettings,
    pub server: ServerSettings,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            models: ModelRegistry::default(),
            inference: InferenceSettings::default(),
            server: ServerSettings::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelRegistry {
    pub active: String,
    pub entries: BTreeMap<String, ModelEntry>,
}

impl Default for ModelRegistry {
    fn default() -> Self {
        let mut entries = BTreeMap::new();
        entries.insert(DEFAULT_MODEL_ID.to_string(), ModelEntry::default());
        Self {
            active: DEFAULT_MODEL_ID.to_string(),
            entries,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelEntry {
    pub config: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub weights: Option<PathBuf>,
}

impl Default for ModelEntry {
    fn default() -> Self {
        Self {
            config: None,
            tokenizer: None,
            weights: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct InferenceSettings {
    pub device: DeviceKind,
    pub precision: Option<Precision>,
    pub template: String,
    pub base_size: u32,
    pub image_size: u32,
    pub crop_mode: bool,
    pub max_new_tokens: usize,
    pub use_cache: bool,
    /// Fraction of GPU memory to use for model + cache (0.0 - 1.0)
    pub gpu_memory_utilization: Option<f32>,
    /// Maximum number of concurrent sequences/batches
    pub max_num_seqs: Option<usize>,
}

impl Default for InferenceSettings {
    fn default() -> Self {
        Self {
            device: DeviceKind::Cpu,
            precision: None,
            template: "plain".to_string(),
            base_size: 1024,
            image_size: 640,
            crop_mode: true,
            max_new_tokens: 512,
            use_cache: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerSettings {
    pub host: String,
    pub port: u16,
    pub model_id: String,
}

impl Default for ServerSettings {
    fn default() -> Self {
        Self {
            host: "0.0.0.0".to_string(),
            port: 8000,
            model_id: DEFAULT_MODEL_ID.to_string(),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ResourceLocation {
    Virtual(VirtualPath),
    Physical(PathBuf),
}

impl ResourceLocation {
    pub fn display_with(&self, fs: &impl VirtualFileSystem) -> Result<String> {
        match self {
            ResourceLocation::Virtual(path) => {
                fs.with_physical_path(path, |p| Ok(p.display().to_string()))
            }
            ResourceLocation::Physical(path) => Ok(path.display().to_string()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModelResources {
    pub config: ResourceLocation,
    pub tokenizer: ResourceLocation,
    pub weights: ResourceLocation,
}

pub struct ConfigDescriptor {
    pub location: ResourceLocation,
}

impl AppConfig {
    pub fn load_or_init(
        fs: &impl VirtualFileSystem,
        override_path: Option<&Path>,
    ) -> Result<(Self, ConfigDescriptor)> {
        match override_path {
            Some(path) => load_physical_config(fs, path),
            None => load_virtual_config(fs),
        }
    }

    pub fn load_with_overrides(
        fs: &impl VirtualFileSystem,
        overrides: ConfigOverrides,
    ) -> Result<(Self, ConfigDescriptor, ModelResources)> {
        let config_path_override = overrides.config_path.clone();
        let (mut config, descriptor) = Self::load_or_init(fs, config_path_override.as_deref())?;
        config += overrides;
        config.normalise(fs)?;
        let resources = config.active_model_resources(fs)?;
        Ok((config, descriptor, resources))
    }

    pub fn normalise(&mut self, fs: &impl VirtualFileSystem) -> Result<()> {
        if self.models.entries.is_empty() {
            self.models
                .entries
                .insert(DEFAULT_MODEL_ID.to_string(), ModelEntry::default());
        }
        if !self.models.entries.contains_key(&self.models.active) {
            self.models
                .entries
                .insert(self.models.active.clone(), ModelEntry::default());
        }

        for (model_id, entry) in self.models.entries.iter_mut() {
            entry.normalise(fs, model_id)?;
        }
        Ok(())
    }

    pub fn active_model_resources(&self, fs: &impl VirtualFileSystem) -> Result<ModelResources> {
        self.model_resources(fs, &self.models.active)
    }

    pub fn model_resources(
        &self,
        _fs: &impl VirtualFileSystem,
        model_id: &str,
    ) -> Result<ModelResources> {
        let entry = self
            .models
            .entries
            .get(model_id)
            .ok_or_else(|| anyhow!("model `{model_id}` not found in configuration"))?;
        Ok(entry.resolved(model_id))
    }

    pub fn apply_overrides(&mut self, overrides: &ConfigOverrides) {
        if let Some(model_id) = overrides.model_id.as_ref() {
            self.models.active = model_id.clone();
            self.models
                .entries
                .entry(model_id.clone())
                .or_insert_with(ModelEntry::default);
        }

        if let Some(entry) = self.models.entries.get_mut(&self.models.active) {
            if let Some(path) = overrides.model_config.as_ref() {
                entry.config = Some(path.clone());
            }
            if let Some(path) = overrides.tokenizer.as_ref() {
                entry.tokenizer = Some(path.clone());
            }
            if let Some(path) = overrides.weights.as_ref() {
                entry.weights = Some(path.clone());
            }
        }

        if let Some(device) = overrides.inference.device {
            self.inference.device = device;
        }
        if overrides.inference.precision.is_some() {
            self.inference.precision = overrides.inference.precision;
        }
        if let Some(template) = overrides.inference.template.as_ref() {
            self.inference.template = template.clone();
        }
        if let Some(base_size) = overrides.inference.base_size {
            self.inference.base_size = base_size;
        }
        if let Some(image_size) = overrides.inference.image_size {
            self.inference.image_size = image_size;
        }
        if let Some(crop_mode) = overrides.inference.crop_mode {
            self.inference.crop_mode = crop_mode;
        }
        if let Some(max_new_tokens) = overrides.inference.max_new_tokens {
            self.inference.max_new_tokens = max_new_tokens;
        }
        if let Some(use_cache) = overrides.inference.use_cache {
            self.inference.use_cache = use_cache;
        }
        if overrides.inference.gpu_memory_utilization.is_some() {
            self.inference.gpu_memory_utilization = overrides.inference.gpu_memory_utilization;
        }
        if overrides.inference.max_num_seqs.is_some() {
            self.inference.max_num_seqs = overrides.inference.max_num_seqs;
        }
        if let Some(host) = overrides.server.host.as_ref() {
            self.server.host = host.clone();
        }
        if let Some(port) = overrides.server.port {
            self.server.port = port;
        }
        if let Some(model_id) = overrides.server.model_id.as_ref() {
            self.server.model_id = model_id.clone();
        }
    }
}

impl ModelEntry {
    fn normalise(&mut self, fs: &impl VirtualFileSystem, model_id: &str) -> Result<()> {
        let model_dir = VirtualPath::model_dir(model_id.to_string());
        fs.ensure_dir(&model_dir)?;
        fs.ensure_parent(&VirtualPath::model_config(model_id.to_string()))?;
        fs.ensure_parent(&VirtualPath::model_tokenizer(model_id.to_string()))?;
        fs.ensure_parent(&VirtualPath::model_weights(model_id.to_string()))?;
        Ok(())
    }

    fn resolved(&self, model_id: &str) -> ModelResources {
        let config = match &self.config {
            Some(path) => ResourceLocation::Physical(path.clone()),
            None => ResourceLocation::Virtual(VirtualPath::model_config(model_id.to_string())),
        };
        let tokenizer = match &self.tokenizer {
            Some(path) => ResourceLocation::Physical(path.clone()),
            None => ResourceLocation::Virtual(VirtualPath::model_tokenizer(model_id.to_string())),
        };
        let weights = match &self.weights {
            Some(path) => ResourceLocation::Physical(path.clone()),
            None => ResourceLocation::Virtual(VirtualPath::model_weights(model_id.to_string())),
        };
        ModelResources {
            config,
            tokenizer,
            weights,
        }
    }
}

fn load_virtual_config(fs: &impl VirtualFileSystem) -> Result<(AppConfig, ConfigDescriptor)> {
    let path = VirtualPath::config_file();
    if !fs.exists(&path)? {
        let mut cfg = AppConfig::default();
        cfg.normalise(fs)?;
        let serialized = toml::to_string_pretty(&cfg)?;
        fs.write(&path, serialized.as_bytes())?;
        return Ok((
            cfg,
            ConfigDescriptor {
                location: ResourceLocation::Virtual(path),
            },
        ));
    }

    let bytes = fs.read(&path)?;
    let contents = String::from_utf8(bytes).context("configuration file is not valid UTF-8")?;
    let mut cfg: AppConfig =
        toml::from_str(&contents).context("failed to parse configuration file")?;
    cfg.normalise(fs)?;
    Ok((
        cfg,
        ConfigDescriptor {
            location: ResourceLocation::Virtual(path),
        },
    ))
}

fn load_physical_config(
    fs: &impl VirtualFileSystem,
    path: &Path,
) -> Result<(AppConfig, ConfigDescriptor)> {
    let path_buf = path.to_path_buf();
    if !path.exists() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
        let mut cfg = AppConfig::default();
        cfg.normalise(fs)?;
        let serialized = toml::to_string_pretty(&cfg)?;
        fs::write(&path_buf, serialized)
            .with_context(|| format!("failed to write configuration to {}", path_buf.display()))?;
        return Ok((
            cfg,
            ConfigDescriptor {
                location: ResourceLocation::Physical(path_buf),
            },
        ));
    }

    let contents = fs::read_to_string(&path_buf)
        .with_context(|| format!("failed to read configuration from {}", path_buf.display()))?;
    let mut cfg: AppConfig = toml::from_str(&contents)
        .with_context(|| format!("failed to parse configuration at {}", path_buf.display()))?;
    cfg.normalise(fs)?;
    Ok((
        cfg,
        ConfigDescriptor {
            location: ResourceLocation::Physical(path_buf),
        },
    ))
}

#[derive(Debug, Default, Clone)]
pub struct ConfigOverrides {
    pub config_path: Option<PathBuf>,
    pub model_id: Option<String>,
    pub model_config: Option<PathBuf>,
    pub tokenizer: Option<PathBuf>,
    pub weights: Option<PathBuf>,
    pub inference: InferenceOverride,
    pub server: ServerOverride,
}

#[derive(Debug, Default, Clone)]
pub struct InferenceOverride {
    pub device: Option<DeviceKind>,
    pub precision: Option<Precision>,
    pub template: Option<String>,
    pub base_size: Option<u32>,
    pub image_size: Option<u32>,
    pub crop_mode: Option<bool>,
    pub max_new_tokens: Option<usize>,
    pub use_cache: Option<bool>,
    pub gpu_memory_utilization: Option<f32>,
    pub max_num_seqs: Option<usize>,
}

#[derive(Debug, Default, Clone)]
pub struct ServerOverride {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub model_id: Option<String>,
}

pub trait ConfigOverride {
    fn apply(self, config: &mut AppConfig);
}

impl ConfigOverride for ConfigOverrides {
    fn apply(self, config: &mut AppConfig) {
        config.apply_overrides(&self);
    }
}

impl<'a> ConfigOverride for &'a ConfigOverrides {
    fn apply(self, config: &mut AppConfig) {
        config.apply_overrides(self);
    }
}

impl<O: ConfigOverride> AddAssign<O> for AppConfig {
    fn add_assign(&mut self, rhs: O) {
        rhs.apply(self);
    }
}

pub fn save_config(
    fs: &impl VirtualFileSystem,
    descriptor: &ConfigDescriptor,
    config: &AppConfig,
) -> Result<()> {
    let serialized = toml::to_string_pretty(config)?;
    match &descriptor.location {
        ResourceLocation::Virtual(path) => fs.write(path, serialized.as_bytes()),
        ResourceLocation::Physical(path) => fs::write(path, serialized)
            .with_context(|| format!("failed to write configuration to {}", path.display())),
    }
}
