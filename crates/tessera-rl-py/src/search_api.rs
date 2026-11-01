// SPDX-License-Identifier: MIT OR Apache-2.0

//! The entry points of `tessera_rl._native.Search` (ADR-0043 section 3),
//! and the evaluator bridge: each batch calls the Python evaluator through
//! a callout (ADR-0041), as `evaluator(observations, legal, hashes, n)`
//! with `bytes`, `bytes`, `bytes` and `int`, and reads back a tuple of two
//! `bytes`, the priors (`n * actions` native-byte-order `f64`) and the
//! values (`n` native-byte-order `f64`). A wrong result type or length is
//! `EvaluatorError::Shape`; a Python exception is `EvaluatorError::Failed`
//! and is kept to be raised again after the tree is rolled back.

use crate::api::Kwargs;
use crate::env::BindError;
use crate::errors::{Kind, error, to_py};
use crate::ffi::capi::{self, CResult, Gil, Mine, Raised, Ref};
use crate::ffi::inner::Inner;
use crate::search::{BatchEval, RawBatch, SearchArgs, SearchOps};
use tessera_rl_search::{EvaluatorError, Eviction};

/// The state behind a `Search` object.
pub(crate) type SearchInner = Inner<dyn SearchOps>;

const SEARCH_KEYS: [&str; 12] = [
    "env_id",
    "capacity",
    "batch",
    "eviction",
    "c_puct",
    "fpu",
    "virtual_loss",
    "batch_size",
    "simulations",
    "noise_alpha",
    "noise_epsilon",
    "seed",
];

/// `make_search(**kwargs)`: a PUCT search object.
pub(crate) fn make_search(g: Gil, kwargs: Option<Ref<'_>>) -> CResult<SearchInner> {
    let kw = Kwargs::parse(g, kwargs)?;
    kw.check(g, &[&SEARCH_KEYS])?;
    let mut args = SearchArgs::new();
    if let Some(v) = kw.u32(g, "capacity")? {
        args.capacity = v;
    }
    if let Some(v) = kw.u32(g, "batch")? {
        args.batch = v;
    }
    args.eviction = match kw.string(g, "eviction")?.as_deref() {
        None | Some("strict") => Eviction::Strict,
        Some("oldest") => Eviction::Oldest,
        Some(other) => {
            return Err(error(
                g,
                Kind::Argument,
                format!("eviction must be \"strict\" or \"oldest\", not {other:?}"),
            ));
        }
    };
    if let Some(v) = kw.f64(g, "c_puct")? {
        args.puct.c_puct = v;
    }
    if let Some(v) = kw.f64(g, "fpu")? {
        args.puct.fpu = v;
    }
    if let Some(v) = kw.u32(g, "virtual_loss")? {
        args.puct.virtual_loss = v;
    }
    if let Some(v) = kw.u32(g, "batch_size")? {
        args.puct.batch_size = v;
    }
    if let Some(v) = kw.u32(g, "simulations")? {
        args.puct.simulations = v;
    }
    if let Some(v) = kw.u64(g, "seed")? {
        args.seed = v;
    }
    match (kw.f64(g, "noise_alpha")?, kw.f64(g, "noise_epsilon")?) {
        (Some(alpha), Some(epsilon)) => args = args.with_noise(alpha, epsilon),
        (None, None) => {}
        _ => {
            return Err(error(
                g,
                Kind::Argument,
                "noise_alpha and noise_epsilon go together".to_owned(),
            ));
        }
    }
    let env_id = kw
        .string(g, "env_id")?
        .unwrap_or_else(|| "tessera/Connect4-v1".to_owned());
    let ops = match env_id.as_str() {
        "tessera/Connect4-v1" => crate::search::connect4(&args),
        other => Err(BindError::Argument(format!(
            "no search for {other:?}; known: [\"tessera/Connect4-v1\"]"
        ))),
    }
    .map_err(|e| to_py(g, e))?;
    Ok(Inner::new(ops, false))
}

/// `layout()`: `{"actions": int, "obs_bytes": int}`.
pub(crate) fn layout(g: Gil, inner: &SearchInner) -> CResult<Mine> {
    inner.with(g, |ops| {
        let d = capi::dict(g)?;
        capi::dict_set(g, &d, c"actions", &capi::int_u64(g, ops.actions().into())?)?;
        let ob = u64::try_from(ops.obs_bytes()).unwrap_or(u64::MAX);
        capi::dict_set(g, &d, c"obs_bytes", &capi::int_u64(g, ob)?)?;
        Ok(d)
    })
}

/// `set_root(record: bytes)`.
pub(crate) fn set_root(g: Gil, inner: &SearchInner, args: Ref<'_>) -> CResult<Mine> {
    let [record] = one_arg(g, args, "set_root(record)")?;
    let bytes = capi::bytes_of(g, record, "record")?;
    inner.with(g, |ops| ops.set_root(bytes).map_err(|e| to_py(g, e)))?;
    Ok(capi::none(g))
}

/// The Python evaluator of one `run` call.
struct PyEval<'a> {
    g: Gil,
    callable: Ref<'a>,
    /// A Python exception raised by the callable, raised again after the
    /// rollback.
    pending: Option<capi::Pending>,
}

fn shape(what: String) -> EvaluatorError {
    EvaluatorError::Shape { what }
}

/// Reads `n` native-byte-order `f64` from `bytes` into `out`.
fn read_f64(bytes: &[u8], out: &mut [f64], what: &str) -> Result<(), EvaluatorError> {
    if bytes.len() != out.len() * 8 {
        return Err(shape(format!(
            "{what}: {} bytes for {} float64 values",
            bytes.len(),
            out.len()
        )));
    }
    // Native byte order: the adapter sends the evaluator's float64 buffer
    // as it is.
    let (chunks, _) = bytes.as_chunks::<8>();
    for (o, chunk) in out.iter_mut().zip(chunks) {
        *o = f64::from_ne_bytes(*chunk);
    }
    Ok(())
}

impl PyEval<'_> {
    fn call(&mut self, batch: &RawBatch<'_>) -> CResult<Mine> {
        let g = self.g;
        // Pending signals (Ctrl-C) between batches: a handler that raises
        // aborts the call like an evaluator exception.
        capi::check_signals(g)?;
        let legal: Vec<u8> = batch.legal.iter().map(|&l| u8::from(l)).collect();
        let hashes: Vec<u8> = batch.hashes.iter().flat_map(|h| h.to_le_bytes()).collect();
        let n = u64::try_from(batch.n).unwrap_or(u64::MAX);
        let args = capi::tuple(
            g,
            vec![
                capi::bytes(g, batch.observations)?,
                capi::bytes(g, &legal)?,
                capi::bytes(g, &hashes)?,
                capi::int_u64(g, n)?,
            ],
        )?;
        capi::call(g, self.callable, &args)
    }
}

impl BatchEval for PyEval<'_> {
    fn eval(
        &mut self,
        batch: &RawBatch<'_>,
        priors: &mut [f64],
        values: &mut [f64],
    ) -> Result<(), EvaluatorError> {
        let Ok(result) = self.call(batch) else {
            self.pending = Some(capi::fetch_pending());
            return Err(EvaluatorError::Failed {
                reason: "the Python evaluator raised".to_owned(),
            });
        };
        let g = self.g;
        let Ok(items) = capi::tuple_items(g, result.as_ref(), "evaluator result") else {
            capi::clear_pending(g);
            return Err(shape(
                "the evaluator must return (priors, values)".to_owned(),
            ));
        };
        let [p, v] = items.as_slice() else {
            // The adapter reports a malformed response as a 1-tuple with
            // its reason.
            let reason = match items.as_slice() {
                [only] => capi::bytes_of(g, *only, "reason").map_or_else(
                    |_| {
                        capi::clear_pending(g);
                        "malformed response".to_owned()
                    },
                    |b| String::from_utf8_lossy(b).into_owned(),
                ),
                _ => format!("{} items, not (priors, values)", items.len()),
            };
            return Err(shape(reason));
        };
        let (Ok(p), Ok(v)) = (
            capi::bytes_of(g, *p, "priors"),
            capi::bytes_of(g, *v, "values"),
        ) else {
            capi::clear_pending(g);
            return Err(shape("priors and values must be bytes".to_owned()));
        };
        read_f64(p, priors, "priors")?;
        read_f64(v, values, "values")
    }
}

/// `run(evaluator)`: one search call.
pub(crate) fn run(g: Gil, inner: &SearchInner, args: Ref<'_>) -> CResult<Mine> {
    let [callable] = one_arg(g, args, "run(evaluator)")?;
    let mut eval = PyEval {
        g,
        callable,
        pending: None,
    };
    let result = inner.with(g, |ops| Ok(ops.run(&mut eval)))?;
    match (result, eval.pending.take()) {
        (Ok(()), _) => Ok(capi::none(g)),
        // The callable's own exception: raised again, the tree rolled back.
        (Err(_), Some(pending)) => {
            capi::restore_pending(pending);
            Err(Raised)
        }
        (Err(e), None) => Err(to_py(g, e)),
    }
}

/// `root_visits() -> list[int]`.
pub(crate) fn root_visits(g: Gil, inner: &SearchInner) -> CResult<Mine> {
    inner.with(g, |ops| {
        let list = capi::list(g)?;
        for v in ops.root_visits() {
            capi::list_append(g, &list, &capi::int_u64(g, v.into())?)?;
        }
        Ok(list)
    })
}

/// `select_action(temperature: float) -> int`.
pub(crate) fn select_action(g: Gil, inner: &SearchInner, args: Ref<'_>) -> CResult<Mine> {
    let [t] = one_arg(g, args, "select_action(temperature)")?;
    let t = capi::to_f64(g, t, "temperature")?;
    let a = inner.with(g, |ops| ops.select_action(t).map_err(|e| to_py(g, e)))?;
    capi::int_u64(g, a.into())
}

/// `calls() -> int`.
pub(crate) fn calls(g: Gil, inner: &SearchInner) -> CResult<Mine> {
    let n = inner.with(g, |ops| Ok(ops.calls()))?;
    capi::int_u64(g, n)
}

/// `tree_hash() -> int`.
pub(crate) fn tree_hash(g: Gil, inner: &SearchInner) -> CResult<Mine> {
    let h = inner.with(g, |ops| Ok(ops.tree_hash()))?;
    capi::int_u64(g, h)
}

/// The single positional argument of `signature`.
fn one_arg<'a>(g: Gil, args: Ref<'a>, signature: &str) -> CResult<[Ref<'a>; 1]> {
    let items = capi::tuple_items(g, args, signature)?;
    <[Ref<'a>; 1]>::try_from(items)
        .map_err(|_| error(g, Kind::Argument, format!("expected {signature}")))
}
