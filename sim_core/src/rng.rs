//! Deterministic RNG derivation. One run seed produces one independent, reproducible
//! stream per agent (and a separate one for market-level randomness), so results
//! depend only on (run_seed, config) -- never on agent iteration order, thread
//! scheduling, etc.

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::types::AgentId;

/// Splits a single run seed into an independent stream per agent using SplitMix64-style
/// mixing, then seeds a ChaCha8 RNG from it. ChaCha8 is used (not thread_rng/OS rng)
/// specifically for cross-platform, cross-run reproducibility given the same seed.
pub fn agent_rng(run_seed: u64, agent_id: AgentId) -> ChaCha8Rng {
    let mixed = splitmix64_derive(run_seed, agent_id as u64 ^ 0x9E37_79B9_7F4A_7C15);
    ChaCha8Rng::seed_from_u64(mixed)
}

/// RNG for market-level randomness (fundamental value process, exogenous shocks, etc.),
/// kept separate from any single agent's stream.
pub fn market_rng(run_seed: u64) -> ChaCha8Rng {
    let mixed = splitmix64_derive(run_seed, 0xD1B5_4A32_D192_ED03);
    ChaCha8Rng::seed_from_u64(mixed)
}

fn splitmix64_derive(seed: u64, salt: u64) -> u64 {
    let mut z = seed.wrapping_add(salt).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
