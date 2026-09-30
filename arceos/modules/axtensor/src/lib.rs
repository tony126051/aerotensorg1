#![no_std]

pub mod zero_copy_loader;
pub mod transformer;
pub mod safetensors;
pub mod nn;
pub mod kv_cache;
pub mod models;
pub mod registry;
pub mod benchmark;

pub use zero_copy_loader::{ModelBlobHeader, ModelContext};
pub use transformer::LayerParallelExecutor;
pub use safetensors::{SafeTensorEntry, SafeTensors};
pub use nn::{Embedding, Linear, RMSNorm, RotaryEmbedding, SwiGLU};
pub use kv_cache::KVCache;
pub use models::{LlamaConfig, LlamaDecoderLayer, LlamaLayerScratch, LlamaModel};
pub use registry::{ModelDescriptor, MultiModelInstance, MultiModelRegistry};
pub use benchmark::{run_system_benchmark, SystemBenchmarkReport};
