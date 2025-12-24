use stwo::core::fields::m31::M31;

/// Number of field elements used for the secret key.
pub const SECRET_KEY_SIZE: usize = 8;

/// Secret key for the PRF (Poseidon2).
/// The key is committed via Poseidon2 hash, and only the commitment is public.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SecretKey(pub [M31; SECRET_KEY_SIZE]);

impl SecretKey {
    /// Create a new secret key from field elements.
    pub fn new(elements: [M31; SECRET_KEY_SIZE]) -> Self {
        Self(elements)
    }

    /// Create a secret key from a u64 seed, for testing.
    pub fn from_seed(seed: u64) -> Self {
        let mut elements = [M31::from(0); SECRET_KEY_SIZE];
        for (i, elem) in elements.iter_mut().enumerate() {
            *elem = M31::from(((seed >> (i * 8)) & 0xFF) as u32);
        }
        Self(elements)
    }

    /// Get the key elements.
    #[inline]
    pub fn elements(&self) -> &[M31; SECRET_KEY_SIZE] {
        &self.0
    }
}

impl Default for SecretKey {
    fn default() -> Self {
        Self([M31::from(0); SECRET_KEY_SIZE])
    }
}

/// Input for unigram watermark detection.
#[derive(Clone, Debug)]
pub struct UnigramInput {
    /// The sequence of tokens to verify.
    pub tokens: Vec<M31>,
    /// The secret key used for PRF computation.
    pub secret_key: SecretKey,
}

impl UnigramInput {
    /// Create new unigram input.
    pub fn new(tokens: Vec<M31>, secret_key: [M31; SECRET_KEY_SIZE]) -> Self {
        Self {
            tokens,
            secret_key: SecretKey::new(secret_key),
        }
    }

    /// Get the number of tokens.
    #[inline]
    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// Get a reference to the tokens.
    #[inline]
    pub fn tokens(&self) -> &[M31] {
        &self.tokens
    }

    /// Get a reference to the secret key.
    #[inline]
    pub fn secret_key(&self) -> &SecretKey {
        &self.secret_key
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_secret_key_from_seed() {
        let key = SecretKey::from_seed(0x123456789ABCDEF0);
        assert_ne!(key.elements()[0], M31::from(0));
    }
}
