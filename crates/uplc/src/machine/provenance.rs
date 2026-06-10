//! Value provenance tracking for the CEK machine.
//!
//! The script's applied arguments (datum / redeemer / script context) are `Data` constants. As the
//! script destructures them with the data-navigation builtins (`unConstrData`, `fstPair`, `sndPair`,
//! `headList`, `tailList`, `unListData`, `unMapData`, `unIData`, `unBData`), we follow that data flow
//! and tag every value it produces with the exact structural path it was reached by. This lets a
//! consumer answer, for each operand of each comparison the script makes, *where in the transaction
//! that value came from* — exactly, and independently of the value itself (so two equal integers
//! reached by different paths never collide).
//!
//! ## How it works without modifying `Value`
//!
//! Provenance is kept in a side table keyed by the **pointer identity** of each value's
//! `Rc<Constant>`. This is robust because:
//! - distinct values are distinct allocations → equal primitives at different paths have different
//!   keys (the whole point);
//! - a value cloned through `lookup_var` / lambda application keeps the same `Rc` pointer, so its
//!   provenance survives binding for free;
//! - we keep a clone of each tagged `Rc<Constant>` alive in the table, so an address can never be
//!   freed and reused while it's registered (no ABA).
//!
//! A value with no entry is, by construction, a literal baked into the script (or derived purely
//! from such literals) — i.e. a `Constant` from the consumer's point of view.

use std::collections::HashMap;
use std::rc::Rc;

use pallas_primitives::alonzo::PlutusData;

use super::runtime::convert_tag_to_constr;
use super::value::{Value, from_pallas_bigint};
use crate::ast::Constant;
use crate::builtins::DefaultFunction;

/// Where a value came from. `Path`/`Derived` are the resolvable provenances a consumer cares about;
/// the remaining variants are intermediate destructuring cursors that only ever attach to the
/// pair/list values produced mid-navigation (never to a comparison operand).
#[derive(Debug, Clone)]
pub enum Provenance {
    /// A value reached at `path` inside applied root `root` (e.g. root "context", path "#0.field2[1]").
    Path { root: Rc<str>, path: String },
    /// Produced by `fun` over its operands. `sources` has one entry per argument, in order — a
    /// nested provenance for tx-derived operands, or a [`Provenance::Literal`] for script constants.
    Derived {
        fun: DefaultFunction,
        sources: Vec<Rc<Provenance>>,
    },
    /// A script constant operand of a `Derived`, kept so the expression renders in full (e.g. the
    /// `5` in `field0 - 5`). `repr` is a short human-readable form of the value.
    Literal { repr: String },
    /// The `(ctor, fields)` pair from `unConstrData` of the constr at `path`.
    ConstrParts {
        root: Rc<str>,
        path: String,
        ctor: i64,
    },
    /// The fields list of the constr at `path`, with `off` elements dropped from the front.
    Fields {
        root: Rc<str>,
        path: String,
        ctor: i64,
        off: usize,
    },
    /// The elements of the list at `path`, with `off` dropped from the front.
    Elements {
        root: Rc<str>,
        path: String,
        off: usize,
    },
    /// The entries of the map at `path`, with `off` dropped from the front.
    Entries {
        root: Rc<str>,
        path: String,
        off: usize,
    },
    /// One map entry (a key/value pair) at `path`.
    EntryPair {
        root: Rc<str>,
        path: String,
        off: usize,
    },
    /// A pair whose components have known provenances directly (not a path). Produced when the script
    /// *builds* a value and then destructures it — `unConstrData(constrData(tag, fields))` — so the
    /// inverse builtins can "see through" the construction and recover what was put in.
    PairAlias {
        fst: Rc<Provenance>,
        snd: Rc<Provenance>,
    },
}

struct Entry {
    // Keeps the allocation alive so its address can't be reused while registered.
    _keepalive: Rc<Constant>,
    prov: Rc<Provenance>,
}

#[derive(Default)]
pub struct ProvenanceTable {
    map: HashMap<usize, Entry>,
}

fn ptr_of(v: &Value) -> Option<usize> {
    match v {
        Value::Con(rc) => Some(Rc::as_ptr(rc) as usize),
        _ => None,
    }
}

/// Structural address segment for a map key. MUST stay byte-for-byte identical to the engine's
/// `key_addr` (`trace_analysis.rs`), since the engine looks paths up by this exact string.
fn key_addr(d: &PlutusData) -> String {
    match d {
        PlutusData::BoundedBytes(b) => format!("[0x{}]", hex::encode::<&[u8]>(b.as_ref())),
        PlutusData::BigInt(b) => format!("[{}]", from_pallas_bigint(b)),
        PlutusData::Constr(c) => format!(
            "[constr#{}]",
            convert_tag_to_constr(c.tag).unwrap_or_else(|| c.any_constructor.unwrap_or(0))
        ),
        PlutusData::Array(_) => "[list]".to_string(),
        PlutusData::Map(_) => "[map]".to_string(),
    }
}

/// Constructor index of a `Data::Constr` value, if `v` is one.
fn constr_index_of(v: &Value) -> Option<i64> {
    if let Value::Con(c) = v {
        if let Constant::Data(PlutusData::Constr(c)) = c.as_ref() {
            let idx =
                convert_tag_to_constr(c.tag).unwrap_or_else(|| c.any_constructor.unwrap_or(0));
            return Some(idx as i64);
        }
    }
    None
}

/// Map-key address of a pair value's first component (used for `Data::Map` entries).
fn pair_key_addr(v: &Value) -> Option<String> {
    if let Value::Con(c) = v {
        if let Constant::ProtoPair(_, _, k, _) = c.as_ref() {
            if let Constant::Data(d) = k.as_ref() {
                return Some(key_addr(d));
            }
        }
    }
    None
}

impl ProvenanceTable {
    /// Tag an applied root constant (its exact `Rc`) as the path origin `root`.
    pub fn seed(&mut self, root: Rc<str>, constant: &Rc<Constant>) {
        self.map.insert(
            Rc::as_ptr(constant) as usize,
            Entry {
                _keepalive: constant.clone(),
                prov: Rc::new(Provenance::Path {
                    root,
                    path: String::new(),
                }),
            },
        );
    }

    pub fn get(&self, v: &Value) -> Option<Rc<Provenance>> {
        ptr_of(v)
            .and_then(|k| self.map.get(&k))
            .map(|e| e.prov.clone())
    }

    fn register(&mut self, v: &Value, prov: Provenance) {
        if let Value::Con(rc) = v {
            self.map.insert(
                Rc::as_ptr(rc) as usize,
                Entry {
                    _keepalive: rc.clone(),
                    prov: Rc::new(prov),
                },
            );
        }
    }

    /// Compute and register the provenance of a builtin's result from its arguments' provenances,
    /// following the data-navigation builtins structurally and tagging everything else as `Derived`.
    pub fn propagate(
        &mut self,
        fun: DefaultFunction,
        args: &[Value],
        result: &Result<Value, super::Error>,
    ) {
        use DefaultFunction::*;

        let Ok(result) = result else {
            return;
        };
        let a0 = args.first();
        let p0 = a0.and_then(|v| self.get(v));

        match fun {
            UnConstrData => match p0.as_deref() {
                Some(Provenance::Path { root, path }) => {
                    if let Some(ctor) = a0.and_then(constr_index_of) {
                        self.register(
                            result,
                            Provenance::ConstrParts {
                                root: root.clone(),
                                path: path.clone(),
                                ctor,
                            },
                        );
                    }
                }
                // See through `unConstrData(constrData(tag, fields))`: recover (tag, fields) directly.
                Some(Provenance::Derived {
                    fun: ConstrData,
                    sources,
                }) if sources.len() >= 2 => {
                    self.register(
                        result,
                        Provenance::PairAlias {
                            fst: sources[0].clone(),
                            snd: sources[1].clone(),
                        },
                    );
                }
                _ => {}
            },
            UnListData => match p0.as_deref() {
                Some(Provenance::Path { root, path }) => {
                    self.register(
                        result,
                        Provenance::Elements {
                            root: root.clone(),
                            path: path.clone(),
                            off: 0,
                        },
                    );
                }
                // See through `unListData(listData(xs))`: the elements are whatever built `xs`.
                Some(Provenance::Derived {
                    fun: ListData,
                    sources,
                }) if !sources.is_empty() => {
                    self.register(result, (*sources[0]).clone());
                }
                _ => {}
            },
            UnMapData => match p0.as_deref() {
                Some(Provenance::Path { root, path }) => {
                    self.register(
                        result,
                        Provenance::Entries {
                            root: root.clone(),
                            path: path.clone(),
                            off: 0,
                        },
                    );
                }
                Some(Provenance::Derived {
                    fun: MapData,
                    sources,
                }) if !sources.is_empty() => {
                    self.register(result, (*sources[0]).clone());
                }
                _ => {}
            },
            FstPair => match p0.as_deref() {
                Some(Provenance::ConstrParts { root, path, .. }) => {
                    self.register(
                        result,
                        Provenance::Path {
                            root: root.clone(),
                            path: format!("{path}.constr"),
                        },
                    );
                }
                Some(Provenance::EntryPair { root, path, .. }) => {
                    let addr = a0
                        .and_then(pair_key_addr)
                        .unwrap_or_else(|| "[?]".to_string());
                    self.register(
                        result,
                        Provenance::Path {
                            root: root.clone(),
                            path: format!("{path}{addr}.key"),
                        },
                    );
                }
                Some(Provenance::PairAlias { fst, .. }) => {
                    self.register(result, (**fst).clone());
                }
                // See through `fstPair(mkPairData(a, b))` → a.
                Some(Provenance::Derived {
                    fun: MkPairData,
                    sources,
                }) if !sources.is_empty() => {
                    self.register(result, (*sources[0]).clone());
                }
                _ => {}
            },
            SndPair => match p0.as_deref() {
                Some(Provenance::ConstrParts { root, path, ctor }) => {
                    self.register(
                        result,
                        Provenance::Fields {
                            root: root.clone(),
                            path: path.clone(),
                            ctor: *ctor,
                            off: 0,
                        },
                    );
                }
                Some(Provenance::EntryPair { root, path, .. }) => {
                    let addr = a0
                        .and_then(pair_key_addr)
                        .unwrap_or_else(|| "[?]".to_string());
                    self.register(
                        result,
                        Provenance::Path {
                            root: root.clone(),
                            path: format!("{path}{addr}"),
                        },
                    );
                }
                Some(Provenance::PairAlias { snd, .. }) => {
                    self.register(result, (**snd).clone());
                }
                // See through `sndPair(mkPairData(a, b))` → b.
                Some(Provenance::Derived {
                    fun: MkPairData,
                    sources,
                }) if sources.len() >= 2 => {
                    self.register(result, (*sources[1]).clone());
                }
                _ => {}
            },
            HeadList => match p0.as_deref() {
                Some(Provenance::Fields {
                    root,
                    path,
                    ctor,
                    off,
                }) => {
                    self.register(
                        result,
                        Provenance::Path {
                            root: root.clone(),
                            path: format!("{path}#{ctor}.field{off}"),
                        },
                    );
                }
                Some(Provenance::Elements { root, path, off }) => {
                    self.register(
                        result,
                        Provenance::Path {
                            root: root.clone(),
                            path: format!("{path}[{off}]"),
                        },
                    );
                }
                Some(Provenance::Entries { root, path, off }) => {
                    self.register(
                        result,
                        Provenance::EntryPair {
                            root: root.clone(),
                            path: path.clone(),
                            off: *off,
                        },
                    );
                }
                // See through `headList(mkCons(x, xs))` → x.
                Some(Provenance::Derived {
                    fun: MkCons,
                    sources,
                }) if !sources.is_empty() => {
                    self.register(result, (*sources[0]).clone());
                }
                _ => {}
            },
            TailList => match p0.as_deref() {
                Some(Provenance::Fields {
                    root,
                    path,
                    ctor,
                    off,
                }) => {
                    self.register(
                        result,
                        Provenance::Fields {
                            root: root.clone(),
                            path: path.clone(),
                            ctor: *ctor,
                            off: off + 1,
                        },
                    );
                }
                Some(Provenance::Elements { root, path, off }) => {
                    self.register(
                        result,
                        Provenance::Elements {
                            root: root.clone(),
                            path: path.clone(),
                            off: off + 1,
                        },
                    );
                }
                Some(Provenance::Entries { root, path, off }) => {
                    self.register(
                        result,
                        Provenance::Entries {
                            root: root.clone(),
                            path: path.clone(),
                            off: off + 1,
                        },
                    );
                }
                // See through `tailList(mkCons(x, xs))` → xs.
                Some(Provenance::Derived {
                    fun: MkCons,
                    sources,
                }) if sources.len() >= 2 => {
                    self.register(result, (*sources[1]).clone());
                }
                _ => {}
            },
            UnIData | UnBData => match p0.as_deref() {
                Some(Provenance::Path { root, path }) => {
                    self.register(
                        result,
                        Provenance::Path {
                            root: root.clone(),
                            path: path.clone(),
                        },
                    );
                }
                // See through `unIData(iData(x))` / `unBData(bData(x))` → x.
                Some(Provenance::Derived {
                    fun: IData | BData,
                    sources,
                }) if !sources.is_empty() => {
                    self.register(result, (*sources[0]).clone());
                }
                _ => {}
            },
            // Selection builtins (`ifThenElse`, `chooseList`, `chooseData`, `chooseUnit`, …) return
            // one of their arguments UNCHANGED — the result is the same `Rc`, so its provenance
            // already rides on it. Overwriting it with a `Derived` here would erase a clean `Path`
            // and break navigation into the selected value (e.g. finding the spent input by its
            // out-ref, then reading its address credential). Detect the pass-through by pointer
            // identity and leave the value's provenance intact.
            _ if ptr_of(result).is_some() && args.iter().any(|a| ptr_of(a) == ptr_of(result)) => {
                // identity pass-through — provenance already correct on this Rc.
            }
            // Everything else genuinely computes a NEW value. If ANY operand carries provenance, the
            // result is derived: record every operand in order — its provenance if it has one, else
            // a Literal of its value — so the expression renders in full (e.g. `field0 - 5`).
            _ => {
                if args.iter().any(|a| self.get(a).is_some()) {
                    let sources: Vec<Rc<Provenance>> = args
                        .iter()
                        .map(|a| {
                            self.get(a).unwrap_or_else(|| {
                                Rc::new(Provenance::Literal {
                                    repr: render_value(a),
                                })
                            })
                        })
                        .collect();
                    self.register(result, Provenance::Derived { fun, sources });
                }
            }
        }
    }
}

/// Short human-readable form of a constant operand value (for `Provenance::Literal`).
fn render_value(v: &Value) -> String {
    if let Value::Con(c) = v {
        match c.as_ref() {
            Constant::Integer(i) => i.to_string(),
            Constant::ByteString(b) => format!("0x{}", hex::encode(b)),
            Constant::Bool(b) => b.to_string(),
            Constant::Unit => "()".to_string(),
            Constant::String(s) => format!("{s:?}"),
            _ => "?".to_string(),
        }
    } else {
        "?".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::Type;
    use num_bigint::BigInt;

    fn con(c: Constant) -> Value {
        Value::Con(Rc::new(c))
    }

    /// Navigating `redeemer = Constr0[42]` via unConstrData → sndPair → headList → unIData yields a
    /// value whose provenance is the exact path `redeemer #0.field0`; and a *separate* literal `42`
    /// (a different allocation) carries no provenance — proving equal values don't collide.
    #[test]
    fn follows_navigation_to_exact_path_without_value_collision() {
        let mut table = ProvenanceTable::default();

        // redeemer root: Constr0[42] (CBOR d8799f182aff).
        let root_pd = crate::plutus_data(&hex::decode("d8799f182aff").unwrap()).unwrap();
        let root_rc = Rc::new(Constant::Data(root_pd));
        let root_val = Value::Con(root_rc.clone());
        table.seed(Rc::from("redeemer"), &root_rc);

        // unConstrData(redeemer) -> (ctor, fields) pair
        let pair = con(Constant::ProtoPair(
            Type::Integer,
            Type::List(Rc::new(Type::Data)),
            Rc::new(Constant::Integer(BigInt::from(0))),
            Rc::new(Constant::ProtoList(Type::Data, vec![])),
        ));
        table.propagate(
            DefaultFunction::UnConstrData,
            &[root_val],
            &Ok(pair.clone()),
        );

        // sndPair(pair) -> fields list
        let fields = con(Constant::ProtoList(Type::Data, vec![]));
        table.propagate(DefaultFunction::SndPair, &[pair], &Ok(fields.clone()));

        // headList(fields) -> field0 (Data)
        let field0 = con(Constant::Integer(BigInt::from(0))); // content irrelevant to propagation
        table.propagate(DefaultFunction::HeadList, &[fields], &Ok(field0.clone()));

        // unIData(field0) -> 42
        let navigated_42 = con(Constant::Integer(BigInt::from(42)));
        table.propagate(
            DefaultFunction::UnIData,
            &[field0],
            &Ok(navigated_42.clone()),
        );

        match table.get(&navigated_42).as_deref() {
            Some(Provenance::Path { root, path }) => {
                assert_eq!(&**root, "redeemer");
                assert_eq!(path, "#0.field0");
            }
            other => panic!("expected exact Path, got {other:?}"),
        }

        // A different literal 42 (separate allocation) must be untracked — no collision.
        let literal_42 = con(Constant::Integer(BigInt::from(42)));
        assert!(table.get(&literal_42).is_none());
    }

    /// `unConstrData(constrData(tag, mkCons(field, nil)))` must "see through" the round-trip and
    /// recover `field`'s provenance — otherwise navigating a value the script reconstructed (e.g.
    /// the spent input it found) loses its origin and collapses to "constant".
    #[test]
    fn sees_through_construct_then_destruct() {
        let mut table = ProvenanceTable::default();

        // A field with a known origin (pretend it's the whole redeemer for the test).
        let field_rc = Rc::new(Constant::Integer(BigInt::from(7)));
        let field = Value::Con(field_rc.clone());
        table.seed(Rc::from("redeemer"), &field_rc);

        // mkCons(field, nil) -> fields list  (Derived{MkCons, [Path, Literal]})
        let nil = con(Constant::ProtoList(Type::Data, vec![]));
        let fields_list = con(Constant::ProtoList(Type::Data, vec![]));
        table.propagate(
            DefaultFunction::MkCons,
            &[field, nil],
            &Ok(fields_list.clone()),
        );

        // constrData(0, fields_list) -> constr  (Derived{ConstrData, [Literal, Derived{MkCons}]})
        let tag = con(Constant::Integer(BigInt::from(0)));
        let constr = con(Constant::Integer(BigInt::from(0))); // placeholder value; only prov matters
        table.propagate(
            DefaultFunction::ConstrData,
            &[tag, fields_list],
            &Ok(constr.clone()),
        );

        // unConstrData(constr) -> pair  (PairAlias)
        let pair = con(Constant::Integer(BigInt::from(0)));
        table.propagate(DefaultFunction::UnConstrData, &[constr], &Ok(pair.clone()));

        // sndPair(pair) -> recovered fields list
        let recovered_fields = con(Constant::ProtoList(Type::Data, vec![]));
        table.propagate(
            DefaultFunction::SndPair,
            &[pair],
            &Ok(recovered_fields.clone()),
        );

        // headList(recovered_fields) -> field0, which must carry the original field's provenance.
        let field0 = con(Constant::Integer(BigInt::from(7)));
        table.propagate(
            DefaultFunction::HeadList,
            &[recovered_fields],
            &Ok(field0.clone()),
        );

        match table.get(&field0).as_deref() {
            Some(Provenance::Path { root, path }) => {
                assert_eq!(&**root, "redeemer");
                assert_eq!(path, "");
            }
            other => {
                panic!("expected field provenance recovered through round-trip, got {other:?}")
            }
        }
    }

    /// Some scripts (Aiken's compiled `expect` on an inline datum) destructure a constr and then
    /// *rebuild* the (tag, fields) pair with `iData` + `listData` + `mkPairData`, then `sndPair` the
    /// rebuilt pair to read the fields. Provenance must survive that `mkPairData` → `sndPair` so the
    /// fields don't collapse to "constant". Mirrors the live Splash reward-redeemer trace (steps
    /// 59–66, `inputs[0].resolved.datum.field0`).
    #[test]
    fn sees_through_mk_pair_data_then_snd_pair() {
        let mut table = ProvenanceTable::default();

        // A list-typed datum root; unListData gives the fields an `Elements` cursor (what headList
        // needs to emit `[i]` paths) — exactly the cursor the real trace carries at this point.
        let root_rc = Rc::new(Constant::Data(
            crate::plutus_data(&hex::decode("9f00ff").unwrap()).unwrap(),
        ));
        let root = Value::Con(root_rc.clone());
        table.seed(Rc::from("datum"), &root_rc);
        let fields = con(Constant::ProtoList(Type::Data, vec![]));
        table.propagate(DefaultFunction::UnListData, &[root], &Ok(fields.clone()));

        // iData(tag) and listData(fields) — both Derived over their args.
        let tag_data = con(Constant::Integer(BigInt::from(0)));
        table.propagate(
            DefaultFunction::IData,
            &[con(Constant::Integer(BigInt::from(0)))],
            &Ok(tag_data.clone()),
        );
        let fields_data = con(Constant::Integer(BigInt::from(0))); // placeholder; only prov matters
        table.propagate(
            DefaultFunction::ListData,
            &[fields],
            &Ok(fields_data.clone()),
        );

        // mkPairData(iData(tag), listData(fields)) -> rebuilt pair (Derived{MkPairData, [..]})
        let pair = con(Constant::Integer(BigInt::from(0)));
        table.propagate(
            DefaultFunction::MkPairData,
            &[tag_data, fields_data],
            &Ok(pair.clone()),
        );

        // sndPair(pair) -> the listData blob; must carry listData(fields)'s provenance.
        let snd = con(Constant::Integer(BigInt::from(0)));
        table.propagate(DefaultFunction::SndPair, &[pair], &Ok(snd.clone()));

        // unListData(snd) -> fields list again; headList -> field0, which must resolve to the root.
        let recovered = con(Constant::ProtoList(Type::Data, vec![]));
        table.propagate(DefaultFunction::UnListData, &[snd], &Ok(recovered.clone()));
        let field0 = con(Constant::Integer(BigInt::from(0)));
        table.propagate(DefaultFunction::HeadList, &[recovered], &Ok(field0.clone()));

        match table.get(&field0).as_deref() {
            Some(Provenance::Path { root, path }) => {
                assert_eq!(&**root, "datum");
                assert_eq!(path, "[0]");
            }
            other => panic!("expected field0 path recovered through mkPairData, got {other:?}"),
        }
    }

    #[test]
    fn arithmetic_over_a_tx_value_is_derived() {
        let mut table = ProvenanceTable::default();
        let root_rc = Rc::new(Constant::Data(
            crate::plutus_data(&hex::decode("182a").unwrap()).unwrap(),
        ));
        let root_val = Value::Con(root_rc.clone());
        table.seed(Rc::from("redeemer"), &root_rc);
        let amount = con(Constant::Integer(BigInt::from(42)));
        table.propagate(DefaultFunction::UnIData, &[root_val], &Ok(amount.clone()));

        let sum = con(Constant::Integer(BigInt::from(52)));
        table.propagate(
            DefaultFunction::AddInteger,
            &[amount, con(Constant::Integer(BigInt::from(10)))],
            &Ok(sum.clone()),
        );

        match table.get(&sum).as_deref() {
            Some(Provenance::Derived { fun, sources }) => {
                assert_eq!(*fun, DefaultFunction::AddInteger);
                // One entry per operand, in order: the tx-derived value, then the literal 10.
                assert_eq!(sources.len(), 2);
                assert!(matches!(sources[0].as_ref(), Provenance::Path { .. }));
                assert!(
                    matches!(sources[1].as_ref(), Provenance::Literal { repr } if repr == "10")
                );
            }
            other => panic!("expected Derived, got {other:?}"),
        }
    }
}
