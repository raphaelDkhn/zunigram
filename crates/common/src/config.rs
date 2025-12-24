
/// Minimum log size for the trace (must be at least 8 for SIMD and FRI).
pub const MIN_LOG_SIZE: u32 = 8;

/// Calculate the minimum log_size needed for a given number of tokens.
pub fn min_log_size_for_tokens(num_tokens: usize) -> u32 {
    let mut log_size = MIN_LOG_SIZE;
    while (1 << log_size) < num_tokens {
        log_size += 1;
    }
    log_size
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_min_log_size() {
        // MIN_LOG_SIZE is 8, so small values return 8
        assert_eq!(min_log_size_for_tokens(1), 8);
        assert_eq!(min_log_size_for_tokens(16), 8);
        assert_eq!(min_log_size_for_tokens(256), 8);
        assert_eq!(min_log_size_for_tokens(257), 9);
        assert_eq!(min_log_size_for_tokens(512), 9);
    }
}
