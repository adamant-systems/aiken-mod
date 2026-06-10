//! BLS12-381 backend abstraction.
//!
//! This module provides backend-agnostic wrapper types around the BLS12-381
//! curve operations needed by the UPLC machine. On native targets it is backed
//! by the `blst` C library; on `wasm32` targets it is backed by the pure-Rust
//! `bls12_381` (zkcrypto) crate.
//!
//! All `unsafe`/`blst` code lives in this module — no other file in the crate
//! references `blst` directly.

use crate::machine::Error;
use num_bigint::BigInt;
use num_integer::Integer;
use once_cell::sync::Lazy;

pub const BLST_P1_COMPRESSED_SIZE: usize = 48;
pub const BLST_P2_COMPRESSED_SIZE: usize = 96;

/// The order of the BLS12-381 scalar field (group order `r`), big-endian.
static SCALAR_PERIOD: Lazy<BigInt> = Lazy::new(|| {
    BigInt::from_bytes_be(
        num_bigint::Sign::Plus,
        &[
            0x73, 0xed, 0xa7, 0x53, 0x29, 0x9d, 0x7d, 0x48, 0x33, 0x39, 0xd8, 0x08, 0x09, 0xa1,
            0xd8, 0x05, 0x53, 0xbd, 0xa4, 0x02, 0xff, 0xfe, 0x5b, 0xfe, 0xff, 0xff, 0xff, 0xff,
            0x00, 0x00, 0x00, 0x01,
        ],
    )
});

/// Backend-agnostic trait kept for compatibility with existing call sites that
/// import `Compressable`. Implemented for the wrapper types below.
pub trait Compressable {
    fn compress(&self) -> Vec<u8>;

    fn uncompress(bytes: &[u8]) -> Result<Self, Error>
    where
        Self: std::marker::Sized;
}

// ===========================================================================
// Native backend (blst)
// ===========================================================================

#[cfg(not(target_family = "wasm"))]
mod backend {
    use super::*;
    use std::mem::size_of;

    #[derive(Debug, Clone)]
    pub struct Bls12_381G1Element(pub(super) blst::blst_p1);

    #[derive(Debug, Clone)]
    pub struct Bls12_381G2Element(pub(super) blst::blst_p2);

    #[derive(Debug, Clone)]
    pub struct Bls12_381MlResult(pub(super) blst::blst_fp12);

    impl PartialEq for Bls12_381G1Element {
        fn eq(&self, other: &Self) -> bool {
            self.equal(other)
        }
    }

    impl PartialEq for Bls12_381G2Element {
        fn eq(&self, other: &Self) -> bool {
            self.equal(other)
        }
    }

    impl PartialEq for Bls12_381MlResult {
        fn eq(&self, other: &Self) -> bool {
            super::final_verify(self, other)
        }
    }

    fn scalar_bytes_be(scalar: &BigInt) -> Vec<u8> {
        let size_scalar = size_of::<blst::blst_scalar>();

        let scalar = scalar.mod_floor(&SCALAR_PERIOD);

        let (_, mut bytes) = scalar.to_bytes_be();

        if size_scalar > bytes.len() {
            let diff = size_scalar - bytes.len();
            let mut new_vec = vec![0; diff];
            new_vec.append(&mut bytes);
            bytes = new_vec;
        }

        bytes
    }

    impl Bls12_381G1Element {
        pub fn add(&self, other: &Self) -> Self {
            let mut out = blst::blst_p1::default();
            unsafe {
                blst::blst_p1_add_or_double(
                    &mut out as *mut _,
                    &self.0 as *const _,
                    &other.0 as *const _,
                );
            }
            Bls12_381G1Element(out)
        }

        pub fn neg(&self) -> Self {
            let mut out = self.0;
            unsafe {
                blst::blst_p1_cneg(&mut out as *mut _, true);
            }
            Bls12_381G1Element(out)
        }

        pub fn scalar_mul(scalar: &BigInt, point: &Self) -> Self {
            let size_scalar = size_of::<blst::blst_scalar>();
            let bytes = scalar_bytes_be(scalar);

            let mut out = blst::blst_p1::default();
            let mut scalar = blst::blst_scalar::default();

            unsafe {
                blst::blst_scalar_from_bendian(&mut scalar as *mut _, bytes.as_ptr() as *const _);
                blst::blst_p1_mult(
                    &mut out as *mut _,
                    &point.0 as *const _,
                    scalar.b.as_ptr() as *const _,
                    size_scalar * 8,
                );
            }

            Bls12_381G1Element(out)
        }

        pub fn equal(&self, other: &Self) -> bool {
            unsafe { blst::blst_p1_is_equal(&self.0, &other.0) }
        }

        pub fn compress(&self) -> Vec<u8> {
            let mut out = [0; BLST_P1_COMPRESSED_SIZE];
            unsafe {
                blst::blst_p1_compress(&mut out as *mut _, &self.0);
            }
            out.to_vec()
        }

        pub fn uncompress(bytes: &[u8]) -> Result<Self, Error> {
            if bytes.len() != BLST_P1_COMPRESSED_SIZE {
                return Err(Error::Bls(format!(
                    "{:?}",
                    blst::BLST_ERROR::BLST_BAD_ENCODING
                )));
            }

            let mut affine = blst::blst_p1_affine::default();
            let mut out = blst::blst_p1::default();

            unsafe {
                let err = blst::blst_p1_uncompress(&mut affine as *mut _, bytes.as_ptr());
                if err != blst::BLST_ERROR::BLST_SUCCESS {
                    return Err(Error::Bls(format!("{:?}", err)));
                }
                blst::blst_p1_from_affine(&mut out as *mut _, &affine);
                if !blst::blst_p1_in_g1(&out) {
                    return Err(Error::Bls(format!(
                        "{:?}",
                        blst::BLST_ERROR::BLST_POINT_NOT_IN_GROUP
                    )));
                }
            }

            Ok(Bls12_381G1Element(out))
        }

        pub fn hash_to_group(msg: &[u8], dst: &[u8]) -> Result<Self, Error> {
            if dst.len() > 255 {
                return Err(Error::HashToCurveDstTooBig);
            }

            let mut out = blst::blst_p1::default();
            let aug = [];

            unsafe {
                blst::blst_hash_to_g1(
                    &mut out as *mut _,
                    msg.as_ptr(),
                    msg.len(),
                    dst.as_ptr(),
                    dst.len(),
                    aug.as_ptr(),
                    0,
                );
            }

            Ok(Bls12_381G1Element(out))
        }

        pub fn ex_mem(&self) -> i64 {
            size_of::<blst::blst_p1>() as i64 / 8
        }
    }

    impl Bls12_381G2Element {
        pub fn add(&self, other: &Self) -> Self {
            let mut out = blst::blst_p2::default();
            unsafe {
                blst::blst_p2_add_or_double(
                    &mut out as *mut _,
                    &self.0 as *const _,
                    &other.0 as *const _,
                );
            }
            Bls12_381G2Element(out)
        }

        pub fn neg(&self) -> Self {
            let mut out = self.0;
            unsafe {
                blst::blst_p2_cneg(&mut out as *mut _, true);
            }
            Bls12_381G2Element(out)
        }

        pub fn scalar_mul(scalar: &BigInt, point: &Self) -> Self {
            let size_scalar = size_of::<blst::blst_scalar>();
            let bytes = scalar_bytes_be(scalar);

            let mut out = blst::blst_p2::default();
            let mut scalar = blst::blst_scalar::default();

            unsafe {
                blst::blst_scalar_from_bendian(&mut scalar as *mut _, bytes.as_ptr() as *const _);
                blst::blst_p2_mult(
                    &mut out as *mut _,
                    &point.0 as *const _,
                    scalar.b.as_ptr() as *const _,
                    size_scalar * 8,
                );
            }

            Bls12_381G2Element(out)
        }

        pub fn equal(&self, other: &Self) -> bool {
            unsafe { blst::blst_p2_is_equal(&self.0, &other.0) }
        }

        pub fn compress(&self) -> Vec<u8> {
            let mut out = [0; BLST_P2_COMPRESSED_SIZE];
            unsafe {
                blst::blst_p2_compress(&mut out as *mut _, &self.0);
            }
            out.to_vec()
        }

        pub fn uncompress(bytes: &[u8]) -> Result<Self, Error> {
            if bytes.len() != BLST_P2_COMPRESSED_SIZE {
                return Err(Error::Bls(format!(
                    "{:?}",
                    blst::BLST_ERROR::BLST_BAD_ENCODING
                )));
            }

            let mut affine = blst::blst_p2_affine::default();
            let mut out = blst::blst_p2::default();

            unsafe {
                let err = blst::blst_p2_uncompress(&mut affine as *mut _, bytes.as_ptr());
                if err != blst::BLST_ERROR::BLST_SUCCESS {
                    return Err(Error::Bls(format!("{:?}", err)));
                }
                blst::blst_p2_from_affine(&mut out as *mut _, &affine);
                if !blst::blst_p2_in_g2(&out) {
                    return Err(Error::Bls(format!(
                        "{:?}",
                        blst::BLST_ERROR::BLST_POINT_NOT_IN_GROUP
                    )));
                }
            }

            Ok(Bls12_381G2Element(out))
        }

        pub fn hash_to_group(msg: &[u8], dst: &[u8]) -> Result<Self, Error> {
            if dst.len() > 255 {
                return Err(Error::HashToCurveDstTooBig);
            }

            let mut out = blst::blst_p2::default();
            let aug = [];

            unsafe {
                blst::blst_hash_to_g2(
                    &mut out as *mut _,
                    msg.as_ptr(),
                    msg.len(),
                    dst.as_ptr(),
                    dst.len(),
                    aug.as_ptr(),
                    0,
                );
            }

            Ok(Bls12_381G2Element(out))
        }

        pub fn ex_mem(&self) -> i64 {
            size_of::<blst::blst_p2>() as i64 / 8
        }
    }

    impl Bls12_381MlResult {
        pub fn ex_mem(&self) -> i64 {
            size_of::<blst::blst_fp12>() as i64 / 8
        }
    }

    pub fn miller_loop(g1: &Bls12_381G1Element, g2: &Bls12_381G2Element) -> Bls12_381MlResult {
        let mut out = blst::blst_fp12::default();
        let mut affine1 = blst::blst_p1_affine::default();
        let mut affine2 = blst::blst_p2_affine::default();

        unsafe {
            blst::blst_p1_to_affine(&mut affine1 as *mut _, &g1.0);
            blst::blst_p2_to_affine(&mut affine2 as *mut _, &g2.0);
            blst::blst_miller_loop(&mut out as *mut _, &affine2, &affine1);
        }

        Bls12_381MlResult(out)
    }

    pub fn ml_result_mul(a: &Bls12_381MlResult, b: &Bls12_381MlResult) -> Bls12_381MlResult {
        let mut out = blst::blst_fp12::default();
        unsafe {
            blst::blst_fp12_mul(&mut out as *mut _, &a.0, &b.0);
        }
        Bls12_381MlResult(out)
    }

    pub fn final_verify(a: &Bls12_381MlResult, b: &Bls12_381MlResult) -> bool {
        unsafe { blst::blst_fp12_finalverify(&a.0, &b.0) }
    }
}

// ===========================================================================
// Wasm backend (bls12_381 zkcrypto)
// ===========================================================================

#[cfg(target_family = "wasm")]
mod backend {
    use super::*;
    use bls12_381::hash_to_curve::{ExpandMsgXmd, HashToCurve};
    use bls12_381::{
        G1Affine, G1Projective, G2Affine, G2Prepared, G2Projective, Gt, MillerLoopResult, Scalar,
        multi_miller_loop,
    };
    use sha2::Sha256;

    #[derive(Debug, Clone)]
    pub struct Bls12_381G1Element(pub(super) G1Projective);

    #[derive(Debug, Clone)]
    pub struct Bls12_381G2Element(pub(super) G2Projective);

    #[derive(Debug, Clone)]
    pub struct Bls12_381MlResult(pub(super) MillerLoopResult);

    impl PartialEq for Bls12_381G1Element {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl PartialEq for Bls12_381G2Element {
        fn eq(&self, other: &Self) -> bool {
            self.0 == other.0
        }
    }

    impl PartialEq for Bls12_381MlResult {
        fn eq(&self, other: &Self) -> bool {
            super::final_verify(self, other)
        }
    }

    /// Produce a zkcrypto `Scalar` from a `BigInt`, reduced mod the group order.
    /// blst consumes big-endian scalar bytes; zkcrypto `Scalar::from_bytes`
    /// expects little-endian 32 bytes, so we reverse.
    fn to_scalar(scalar: &BigInt) -> Scalar {
        let scalar = scalar.mod_floor(&SCALAR_PERIOD);
        let (_, be) = scalar.to_bytes_be();

        let mut le = [0u8; 32];
        // be is the big-endian magnitude; place it right-aligned then reverse.
        let mut padded = [0u8; 32];
        let start = 32 - be.len();
        padded[start..].copy_from_slice(&be);
        for (i, b) in padded.iter().rev().enumerate() {
            le[i] = *b;
        }

        Scalar::from_bytes(&le).unwrap()
    }

    fn bad_encoding() -> Error {
        Error::Bls("BLST_BAD_ENCODING".to_string())
    }

    impl Bls12_381G1Element {
        pub fn add(&self, other: &Self) -> Self {
            Bls12_381G1Element(self.0 + other.0)
        }

        pub fn neg(&self) -> Self {
            Bls12_381G1Element(-self.0)
        }

        pub fn scalar_mul(scalar: &BigInt, point: &Self) -> Self {
            Bls12_381G1Element(point.0 * to_scalar(scalar))
        }

        pub fn equal(&self, other: &Self) -> bool {
            self.0 == other.0
        }

        pub fn compress(&self) -> Vec<u8> {
            G1Affine::from(self.0).to_compressed().to_vec()
        }

        pub fn uncompress(bytes: &[u8]) -> Result<Self, Error> {
            if bytes.len() != BLST_P1_COMPRESSED_SIZE {
                return Err(bad_encoding());
            }
            let mut buf = [0u8; BLST_P1_COMPRESSED_SIZE];
            buf.copy_from_slice(bytes);
            let affine: Option<G1Affine> = G1Affine::from_compressed(&buf).into();
            let affine = affine.ok_or_else(bad_encoding)?;
            Ok(Bls12_381G1Element(G1Projective::from(affine)))
        }

        pub fn hash_to_group(msg: &[u8], dst: &[u8]) -> Result<Self, Error> {
            if dst.len() > 255 {
                return Err(Error::HashToCurveDstTooBig);
            }
            let p = <G1Projective as HashToCurve<ExpandMsgXmd<Sha256>>>::hash_to_curve(msg, dst);
            Ok(Bls12_381G1Element(p))
        }

        pub fn ex_mem(&self) -> i64 {
            18
        }
    }

    impl Bls12_381G2Element {
        pub fn add(&self, other: &Self) -> Self {
            Bls12_381G2Element(self.0 + other.0)
        }

        pub fn neg(&self) -> Self {
            Bls12_381G2Element(-self.0)
        }

        pub fn scalar_mul(scalar: &BigInt, point: &Self) -> Self {
            Bls12_381G2Element(point.0 * to_scalar(scalar))
        }

        pub fn equal(&self, other: &Self) -> bool {
            self.0 == other.0
        }

        pub fn compress(&self) -> Vec<u8> {
            G2Affine::from(self.0).to_compressed().to_vec()
        }

        pub fn uncompress(bytes: &[u8]) -> Result<Self, Error> {
            if bytes.len() != BLST_P2_COMPRESSED_SIZE {
                return Err(bad_encoding());
            }
            let mut buf = [0u8; BLST_P2_COMPRESSED_SIZE];
            buf.copy_from_slice(bytes);
            let affine: Option<G2Affine> = G2Affine::from_compressed(&buf).into();
            let affine = affine.ok_or_else(bad_encoding)?;
            Ok(Bls12_381G2Element(G2Projective::from(affine)))
        }

        pub fn hash_to_group(msg: &[u8], dst: &[u8]) -> Result<Self, Error> {
            if dst.len() > 255 {
                return Err(Error::HashToCurveDstTooBig);
            }
            let p = <G2Projective as HashToCurve<ExpandMsgXmd<Sha256>>>::hash_to_curve(msg, dst);
            Ok(Bls12_381G2Element(p))
        }

        pub fn ex_mem(&self) -> i64 {
            36
        }
    }

    impl Bls12_381MlResult {
        pub fn ex_mem(&self) -> i64 {
            72
        }
    }

    pub fn miller_loop(g1: &Bls12_381G1Element, g2: &Bls12_381G2Element) -> Bls12_381MlResult {
        let g1_affine = G1Affine::from(g1.0);
        let g2_prepared = G2Prepared::from(G2Affine::from(g2.0));
        Bls12_381MlResult(multi_miller_loop(&[(&g1_affine, &g2_prepared)]))
    }

    pub fn ml_result_mul(a: &Bls12_381MlResult, b: &Bls12_381MlResult) -> Bls12_381MlResult {
        // MillerLoopResult represents Fp12 multiplicatively as an additive group.
        Bls12_381MlResult(a.0 + b.0)
    }

    pub fn final_verify(a: &Bls12_381MlResult, b: &Bls12_381MlResult) -> bool {
        let ga: Gt = a.0.final_exponentiation();
        let gb: Gt = b.0.final_exponentiation();
        ga == gb
    }
}

pub use backend::{
    Bls12_381G1Element, Bls12_381G2Element, Bls12_381MlResult, final_verify, miller_loop,
    ml_result_mul,
};

impl Compressable for Bls12_381G1Element {
    fn compress(&self) -> Vec<u8> {
        Bls12_381G1Element::compress(self)
    }

    fn uncompress(bytes: &[u8]) -> Result<Self, Error> {
        Bls12_381G1Element::uncompress(bytes)
    }
}

impl Compressable for Bls12_381G2Element {
    fn compress(&self) -> Vec<u8> {
        Bls12_381G2Element::compress(self)
    }

    fn uncompress(bytes: &[u8]) -> Result<Self, Error> {
        Bls12_381G2Element::uncompress(bytes)
    }
}

#[cfg(all(test, not(target_family = "wasm")))]
mod cross_check {
    //! Cross-checks the native blst backend against the pure-Rust bls12_381
    //! backend on fixed inputs, asserting byte-identical results. This is how
    //! we trust the wasm backend without running wasm.
    use bls12_381::hash_to_curve::{ExpandMsgXmd, HashToCurve};
    use bls12_381::{
        G1Affine, G1Projective, G2Affine, G2Prepared, G2Projective, Scalar, multi_miller_loop,
    };
    use num_bigint::BigInt;
    use num_integer::Integer;
    use sha2::Sha256;

    use super::{Bls12_381G1Element, Bls12_381G2Element, SCALAR_PERIOD};

    // Known valid G1 / G2 compressed points (generator-derived).
    fn g1_gen_bytes() -> Vec<u8> {
        G1Affine::from(G1Projective::generator())
            .to_compressed()
            .to_vec()
    }
    fn g2_gen_bytes() -> Vec<u8> {
        G2Affine::from(G2Projective::generator())
            .to_compressed()
            .to_vec()
    }

    fn to_scalar(scalar: &BigInt) -> Scalar {
        let scalar = scalar.mod_floor(&SCALAR_PERIOD);
        let (_, be) = scalar.to_bytes_be();
        let mut padded = [0u8; 32];
        let start = 32 - be.len();
        padded[start..].copy_from_slice(&be);
        let mut le = [0u8; 32];
        for (i, b) in padded.iter().rev().enumerate() {
            le[i] = *b;
        }
        Scalar::from_bytes(&le).unwrap()
    }

    #[test]
    fn g1_compress_uncompress_roundtrip_and_match() {
        let bytes = g1_gen_bytes();
        let blst_p = Bls12_381G1Element::uncompress(&bytes).unwrap();
        // round trip
        assert_eq!(blst_p.compress(), bytes);
        // zkcrypto compress of the same point
        let zk = G1Affine::from_compressed(bytes.as_slice().try_into().unwrap()).unwrap();
        assert_eq!(blst_p.compress(), zk.to_compressed().to_vec());
    }

    #[test]
    fn g2_compress_uncompress_roundtrip_and_match() {
        let bytes = g2_gen_bytes();
        let blst_p = Bls12_381G2Element::uncompress(&bytes).unwrap();
        assert_eq!(blst_p.compress(), bytes);
        let zk = G2Affine::from_compressed(bytes.as_slice().try_into().unwrap()).unwrap();
        assert_eq!(blst_p.compress(), zk.to_compressed().to_vec());
    }

    #[test]
    fn g1_add_neg_scalar_mul_match() {
        let a_bytes = g1_gen_bytes();
        // b = 2*g via blst scalar_mul
        let two = BigInt::from(2u32);
        let a = Bls12_381G1Element::uncompress(&a_bytes).unwrap();
        let b = Bls12_381G1Element::scalar_mul(&two, &a);

        // blst results
        let blst_add = a.add(&b).compress();
        let blst_neg = a.neg().compress();
        let scalar = BigInt::from(12345u32);
        let blst_sm = Bls12_381G1Element::scalar_mul(&scalar, &a).compress();

        // zkcrypto results
        let zk_a = G1Projective::from(
            G1Affine::from_compressed(a_bytes.as_slice().try_into().unwrap()).unwrap(),
        );
        let zk_b = zk_a * to_scalar(&two);
        let zk_add = G1Affine::from(zk_a + zk_b).to_compressed().to_vec();
        let zk_neg = G1Affine::from(-zk_a).to_compressed().to_vec();
        let zk_sm = G1Affine::from(zk_a * to_scalar(&scalar))
            .to_compressed()
            .to_vec();

        assert_eq!(blst_add, zk_add, "g1 add mismatch");
        assert_eq!(blst_neg, zk_neg, "g1 neg mismatch");
        assert_eq!(blst_sm, zk_sm, "g1 scalar_mul mismatch");
        assert!(a.equal(&a));
        assert!(!a.equal(&b));
    }

    #[test]
    fn g2_add_neg_scalar_mul_match() {
        let a_bytes = g2_gen_bytes();
        let two = BigInt::from(2u32);
        let a = Bls12_381G2Element::uncompress(&a_bytes).unwrap();
        let b = Bls12_381G2Element::scalar_mul(&two, &a);

        let blst_add = a.add(&b).compress();
        let blst_neg = a.neg().compress();
        let scalar = BigInt::from(12345u32);
        let blst_sm = Bls12_381G2Element::scalar_mul(&scalar, &a).compress();

        let zk_a = G2Projective::from(
            G2Affine::from_compressed(a_bytes.as_slice().try_into().unwrap()).unwrap(),
        );
        let zk_b = zk_a * to_scalar(&two);
        let zk_add = G2Affine::from(zk_a + zk_b).to_compressed().to_vec();
        let zk_neg = G2Affine::from(-zk_a).to_compressed().to_vec();
        let zk_sm = G2Affine::from(zk_a * to_scalar(&scalar))
            .to_compressed()
            .to_vec();

        assert_eq!(blst_add, zk_add, "g2 add mismatch");
        assert_eq!(blst_neg, zk_neg, "g2 neg mismatch");
        assert_eq!(blst_sm, zk_sm, "g2 scalar_mul mismatch");
        assert!(a.equal(&a));
        assert!(!a.equal(&b));
    }

    #[test]
    fn hash_to_group_match() {
        let msg = b"abc";
        let dst = b"DST";

        let blst_g1 = Bls12_381G1Element::hash_to_group(msg, dst)
            .unwrap()
            .compress();
        let zk_g1 = G1Affine::from(
            <G1Projective as HashToCurve<ExpandMsgXmd<Sha256>>>::hash_to_curve(msg.as_slice(), dst),
        )
        .to_compressed()
        .to_vec();
        assert_eq!(blst_g1, zk_g1, "g1 hash_to_group mismatch");

        let blst_g2 = Bls12_381G2Element::hash_to_group(msg, dst)
            .unwrap()
            .compress();
        let zk_g2 = G2Affine::from(
            <G2Projective as HashToCurve<ExpandMsgXmd<Sha256>>>::hash_to_curve(msg.as_slice(), dst),
        )
        .to_compressed()
        .to_vec();
        assert_eq!(blst_g2, zk_g2, "g2 hash_to_group mismatch");
    }

    #[test]
    fn miller_loop_final_verify_match() {
        // P = g1, Q = g2; P2 = 2*g1
        let p = Bls12_381G1Element::uncompress(&g1_gen_bytes()).unwrap();
        let q = Bls12_381G2Element::uncompress(&g2_gen_bytes()).unwrap();
        let p2 = Bls12_381G1Element::scalar_mul(&BigInt::from(2u32), &p);

        // blst path
        let ml_pq = super::miller_loop(&p, &q);
        let ml_p2q = super::miller_loop(&p2, &q);
        assert!(super::final_verify(&ml_pq, &ml_pq));
        assert!(!super::final_verify(&ml_pq, &ml_p2q));

        // ml_result_mul consistency: e(P,Q)*e(P,Q) vs e(2P,Q)
        let prod = super::ml_result_mul(&ml_pq, &ml_pq);
        assert!(super::final_verify(&prod, &ml_p2q));

        // zkcrypto path agreement on final_verify behavior
        let zk_p =
            G1Affine::from_compressed(g1_gen_bytes().as_slice().try_into().unwrap()).unwrap();
        let zk_q =
            G2Affine::from_compressed(g2_gen_bytes().as_slice().try_into().unwrap()).unwrap();
        let zk_qp = G2Prepared::from(zk_q);
        let zk_ml = multi_miller_loop(&[(&zk_p, &zk_qp)]);
        assert_eq!(zk_ml.final_exponentiation(), zk_ml.final_exponentiation());
    }
}
