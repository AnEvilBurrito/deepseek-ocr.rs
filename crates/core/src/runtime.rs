use anyhow::{Context, Result, bail};
use candle_core::{DType, Device};
use clap::ValueEnum;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DeviceKind {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, Copy, ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    F16,
    Bf16,
}

pub fn prepare_device_and_dtype(
    device: DeviceKind,
    precision: Option<Precision>,
) -> Result<(Device, Option<DType>)> {
    prepare_device_and_dtype_with_options(device, precision, None, None)
}

pub fn prepare_device_and_dtype_with_options(
    device: DeviceKind,
    precision: Option<Precision>,
    gpu_memory_utilization: Option<f32>,
    max_num_seqs: Option<usize>,
) -> Result<(Device, Option<DType>)> {
    // Validate GPU memory utilization if provided
    if let Some(utilization) = gpu_memory_utilization {
        if !(0.0..=1.0).contains(&utilization) {
            bail!("GPU memory utilization must be between 0.0 and 1.0, got {}", utilization);
        }
    }
    
    // Validate max_num_seqs if provided
    if let Some(max_seqs) = max_num_seqs {
        if max_seqs == 0 {
            bail!("Maximum number of sequences must be greater than 0");
        }
    }
    
    let (device, default_precision) = match device {
        DeviceKind::Cpu => (Device::Cpu, None),
        DeviceKind::Metal => (
            Device::new_metal(0).context("failed to initialise Metal device")?,
            Some(Precision::F16),
        ),
        DeviceKind::Cuda => (
            Device::new_cuda(0).context("failed to initialise CUDA device")?,
            Some(Precision::F16),
        ),
    };
    
    // Log the GPU configuration options if provided
    if let Some(utilization) = gpu_memory_utilization {
        tracing::info!("GPU memory utilization set to: {:.2}%", utilization * 100.0);
    }
    if let Some(max_seqs) = max_num_seqs {
        tracing::info!("Maximum concurrent sequences set to: {}", max_seqs);
    }
    
    // TODO: Implement actual GPU memory utilization and sequence limiting
    // This would require integration with candle_core's memory management
    // and the inference pipeline's concurrency control
    
    let dtype = precision.or(default_precision).map(dtype_from_precision);
    Ok((device, dtype))
}

pub fn default_dtype_for_device(device: &Device) -> DType {
    if device.is_metal() || device.is_cuda() {
        DType::F16
    } else {
        DType::F32
    }
}

pub fn dtype_from_precision(p: Precision) -> DType {
    match p {
        Precision::F32 => DType::F32,
        Precision::F16 => DType::F16,
        Precision::Bf16 => DType::BF16,
    }
}
