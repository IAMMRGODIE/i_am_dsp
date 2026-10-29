//! The GUI-neutral half of the effects bundle.
//!
//! [processor] is the plumbing every product needs regardless of what draws it: an
//! effect kept inside a `Paramed` so the host and an interface share one parameter map,
//! rebuilt around that same map when the host announces its sample rate, plus the one
//! message the interface sends. There is no iced here, and no plug-in export: the iced
//! bundle and the wry product each bring their own GUI and their own CLAP entry.

pub mod processor;
