//! Small helpers for reading bot-variant parameters out of a generic TOML table,
//! so each bot module can declare its own defaults inline instead of needing a
//! dedicated serde struct + tag per variant.

use toml::Value;

pub fn get_f64(tbl: &Value, key: &str, default: f64) -> f64 {
    tbl.get(key).and_then(|v| v.as_float().or_else(|| v.as_integer().map(|i| i as f64))).unwrap_or(default)
}

pub fn get_u32(tbl: &Value, key: &str, default: u32) -> u32 {
    tbl.get(key).and_then(|v| v.as_integer()).map(|i| i as u32).unwrap_or(default)
}

pub fn get_bool(tbl: &Value, key: &str, default: bool) -> bool {
    tbl.get(key).and_then(|v| v.as_bool()).unwrap_or(default)
}

pub fn get_str<'a>(tbl: &'a Value, key: &str, default: &'a str) -> &'a str {
    tbl.get(key).and_then(|v| v.as_str()).unwrap_or(default)
}

/// Uniform random draw in `[lo, hi]` read from a two-element `[lo, hi]` TOML array,
/// falling back to a fixed default when absent or degenerate.
pub fn get_range(tbl: &Value, key: &str, default: (f64, f64)) -> (f64, f64) {
    if let Some(arr) = tbl.get(key).and_then(|v| v.as_array()) {
        if arr.len() == 2 {
            let lo = arr[0].as_float().or_else(|| arr[0].as_integer().map(|i| i as f64));
            let hi = arr[1].as_float().or_else(|| arr[1].as_integer().map(|i| i as f64));
            if let (Some(lo), Some(hi)) = (lo, hi) {
                return (lo, hi);
            }
        }
    }
    default
}
