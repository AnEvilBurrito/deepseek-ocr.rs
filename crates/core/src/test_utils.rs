use std::{
    path::Path,
    sync::{Arc, Mutex},
};

use anyhow::{anyhow, Context, Result};
use once_cell::sync::OnceCell;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::DynamicImage;

use crate::{
    config::{load_ocr_config, DeepseekV2Config},
    model::{build_global_view, image_to_tensor, DeepseekOcrModel, DEFAULT_WEIGHTS_PATH},
    transformer::{model::DeepseekLanguageModel, weights::TransformerWeights},
};

static OCR_MODEL: OnceCell<Arc<Mutex<DeepseekOcrModel>>> = OnceCell::new();
static LANGUAGE_ASSETS: OnceCell<SharedLanguageAssets> = OnceCell::new();

struct SharedLanguageAssets {
    config: Arc<DeepseekV2Config>,
    transformer: Arc<TransformerWeights>,
    language_model: Arc<Mutex<DeepseekLanguageModel>>,
}

fn load_language_assets() -> Result<SharedLanguageAssets> {
    let weights = Path::new(DEFAULT_WEIGHTS_PATH);
    if !weights.exists() {
        return Err(anyhow!("DeepSeek-OCR weights not present at {:?}", weights));
    }
    let cfg = load_ocr_config(None)
        .context("unable to load OCR config")?
        .resolved_language_config()
        .context("missing language config")?;
    let cfg = Arc::new(cfg);
    let device = Device::Cpu;
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &device)
            .context("failed to mmap language model weights")?
    };
    let model = DeepseekLanguageModel::load(Arc::clone(&cfg), &vb)
        .context("failed to construct language model")?;
    let transformer = model.transformer_weights_arc();
    Ok(SharedLanguageAssets {
        config: cfg,
        transformer,
        language_model: Arc::new(Mutex::new(model)),
    })
}

fn load_ocr_model() -> Result<Arc<Mutex<DeepseekOcrModel>>> {
    let weights = Path::new(DEFAULT_WEIGHTS_PATH);
    if !weights.exists() {
        return Err(anyhow!("DeepSeek-OCR weights not present at {:?}", weights));
    }
    let device = Device::Cpu;
    let model = DeepseekOcrModel::load(None, Some(weights), device, DType::F32)
        .context("failed to load shared DeepseekOcrModel")?;
    Ok(Arc::new(Mutex::new(model)))
}

pub fn shared_ocr_model() -> Result<&'static Arc<Mutex<DeepseekOcrModel>>> {
    OCR_MODEL.get_or_try_init(load_ocr_model)
}

pub fn with_shared_ocr_model<F, T>(op: F) -> Result<T>
where
    F: FnOnce(&DeepseekOcrModel) -> Result<T>,
{
    let model_arc = shared_ocr_model()?;
    let guard = model_arc.lock().expect("ocr model lock poisoned");
    let result = op(&guard);
    drop(guard);
    result
}

pub fn with_shared_language_model<F, T>(op: F) -> Result<T>
where
    F: FnOnce(&DeepseekLanguageModel) -> Result<T>,
{
    let assets = LANGUAGE_ASSETS.get_or_try_init(load_language_assets)?;
    let guard = assets
        .language_model
        .lock()
        .expect("language model lock poisoned");
    op(&guard)
}

pub fn shared_language_config() -> Result<Arc<DeepseekV2Config>> {
    Ok(Arc::clone(
        &LANGUAGE_ASSETS
            .get_or_try_init(load_language_assets)?
            .config,
    ))
}

pub fn shared_transformer_weights() -> Result<Arc<TransformerWeights>> {
    Ok(Arc::clone(
        &LANGUAGE_ASSETS
            .get_or_try_init(load_language_assets)?
            .transformer,
    ))
}

fn load_image(path: &Path) -> Result<DynamicImage> {
    image::ImageReader::open(path)
        .with_context(|| format!("failed to open image at {}", path.display()))?
        .decode()
        .with_context(|| format!("failed to decode image at {}", path.display()))
}

pub fn global_view_tensor_from_path(
    image_path: &Path,
    base_size: u32,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    let image = load_image(image_path)?;
    let global = build_global_view(&image, base_size);
    image_to_tensor(&global, device, dtype)
}

pub fn build_global_view_from_path(image_path: &Path, base_size: u32) -> Result<DynamicImage> {
    let image = load_image(image_path)?;
    Ok(build_global_view(&image, base_size))
}
