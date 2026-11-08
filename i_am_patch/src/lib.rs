//! The patch graph's reusable building blocks.
//!
//! The patcher and the rack share these: the LR4 crossover, the control-rate
//! modulation sources and their runtime. Nothing here knows about a graph, a
//! page or a host yet; those live in the products.

pub mod block;
pub mod compile;
pub mod engine;
pub mod nodes;
pub mod poly;
pub mod crossover;
pub mod graph;
pub mod modulation;
pub mod sources;
pub mod text;
