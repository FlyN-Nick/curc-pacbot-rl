//! Gumbel AlphaZero search (Danihelka et al., "Policy improvement by planning with Gumbel",
//! ICLR 2022; the root and interior rules of DeepMind's mctx), with the real game as the model,
//! batched over many games: each wave runs one simulation in every game, and the leaves of the
//! wave are evaluated by the network in one call.
//!
//! - Root: Gumbel-top-k sampling of the actions to consider, then sequential halving of the
//!   simulation budget among them. The action played is the survivor; the policy target is
//!   softmax(logits + sigma(completed Q)), which improves on the network's policy even with few
//!   simulations.
//! - Interior: the deterministic rule argmax(pi'(a) - N(a) / (1 + sum N)).
//! - Q values: completed with the mixed value estimate for unvisited actions, rescaled to [0, 1]
//!   by the node's min and max, and sigma(q) = (c_visit + max N) * c_scale * q.
//! - Frightened ghosts move randomly: each simulation replays from the root with fresh
//!   randomness (an open-loop tree, whose statistics average over the random outcomes), so the
//!   search doesn't know how the real game's randomness will turn out.
//!
//! With `puct` set, the same batched search uses classic AlphaZero / MuZero rules instead
//! (Silver et al. 2018; Schrittwieser et al. 2020), for comparing the two:
//!
//! - Selection, at the root and inside the tree: argmax Q(a) + P(a) sqrt(N) / (1 + N(a))
//!   (c_puct + ln((N + c_puct_base + 1) / c_puct_base)), with N the node's visits (its children's
//!   plus one) and Q rescaled to [0, 1] by the min and max Q seen in the tree so far, as in
//!   MuZero (rewards are unbounded); unvisited actions get the node's mixed value estimate.
//! - Exploration (gumbel_scale > 0, i.e. training): the root prior is mixed with Dirichlet noise,
//!   (1 - dirichlet_fraction) P + dirichlet_fraction Dir(dirichlet_alpha / legal actions), and
//!   the action played is sampled in proportion to visits^(1 / temperature). With
//!   gumbel_scale = 0 (evaluation) there is no noise and the most visited action is played.
//! - The policy target is the root's visit distribution.
//!
//! Rewards are game rewards times `reward_scale`, discounted by `discount` per move; values and
//! logits come from the evaluator, a Python callable
//! `(obs: float32 (B, 17, 28, 31), legal: bool (B, 5)) -> (values float32 (B,), logits float32 (B, 5))`.

use ndarray::prelude::*;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::prelude::*;
use rand::rngs::SmallRng;
use rand_distr::{Dirichlet, Distribution};
use rayon::prelude::*;

use crate::env::{Action, PacmanGym, PacmanGymConfiguration, OBS_SHAPE};

const A: usize = 5;
const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
#[pyclass(get_all, set_all)]
pub struct GumbelConfig {
    /// Simulations per move (each one network evaluation)
    pub num_simulations: usize,
    /// Discount per move
    pub discount: f32,
    /// Game rewards are multiplied by this (the network's units)
    pub reward_scale: f32,
    /// sigma(q) = (c_visit + max N) * c_scale * q (mctx defaults 50 and 0.1)
    pub c_visit: f32,
    pub c_scale: f32,
    /// Scale of the Gumbel noise at the root: 1 for training (exploration), 0 for evaluation.
    /// With `puct`, any value above 0 turns on PUCT's exploration (Dirichlet noise and sampling)
    pub gumbel_scale: f32,
    /// Simulations stop descending (and bootstrap from the network) this many moves deep
    pub max_depth: usize,
    /// Threads for the search's game work: 1 (the default) runs it on the calling thread, 0
    /// on rayon's global pool. Each simulation's work is small (~10 us per game), so handing it
    /// to other threads mostly adds waiting: 1 thread was fastest on a busy 16-thread machine
    /// (726 moves/s against 318 with 4 and 194 with 8, 152 games, 32 simulations).
    pub threads: usize,
    /// Classic AlphaZero / MuZero search (PUCT) instead of Gumbel's rules (see the module docs)
    pub puct: bool,
    /// PUCT's exploration constant and its slow growth with visits (MuZero's c1 and c2)
    pub c_puct: f32,
    pub c_puct_base: f32,
    /// Total concentration of the root's Dirichlet noise, divided among the legal actions
    /// (AlphaZero's rule of thumb, alpha = 10 / typical number of legal moves)
    pub dirichlet_alpha: f32,
    /// Share of the root prior that is noise, when exploring
    pub dirichlet_fraction: f32,
    /// The played action is sampled in proportion to visits^(1 / temperature), when exploring
    pub temperature: f32,
}

#[pymethods]
impl GumbelConfig {
    #[new]
    #[pyo3(signature = (num_simulations = 32, discount = 0.99, reward_scale = 0.02, c_visit = 50.0, c_scale = 0.1, gumbel_scale = 1.0, max_depth = 64, threads = 1, puct = false, c_puct = 1.25, c_puct_base = 19652.0, dirichlet_alpha = 10.0, dirichlet_fraction = 0.25, temperature = 1.0))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_simulations: usize,
        discount: f32,
        reward_scale: f32,
        c_visit: f32,
        c_scale: f32,
        gumbel_scale: f32,
        max_depth: usize,
        threads: usize,
        puct: bool,
        c_puct: f32,
        c_puct_base: f32,
        dirichlet_alpha: f32,
        dirichlet_fraction: f32,
        temperature: f32,
    ) -> PyResult<Self> {
        if num_simulations < 1 {
            return Err(PyValueError::new_err("num_simulations must be at least 1"));
        }
        if max_depth < 1 {
            return Err(PyValueError::new_err("max_depth must be at least 1"));
        }
        if !(dirichlet_alpha > 0.0) || !(0.0..=1.0).contains(&dirichlet_fraction) {
            return Err(PyValueError::new_err("dirichlet_alpha must be above 0 and dirichlet_fraction in [0, 1]"));
        }
        if !(temperature > 0.0) || !(c_puct_base > 0.0) {
            return Err(PyValueError::new_err("temperature and c_puct_base must be above 0"));
        }
        Ok(Self {
            num_simulations,
            discount,
            reward_scale,
            c_visit,
            c_scale,
            gumbel_scale,
            max_depth,
            threads,
            puct,
            c_puct,
            c_puct_base,
            dirichlet_alpha,
            dirichlet_fraction,
            temperature,
        })
    }

    pub fn __repr__(&self) -> String {
        format!("{self:?}")
    }
}

#[derive(Clone, Debug)]
struct Node {
    legal: [bool; A],
    /// The network's logits (only legal entries are used)
    logits: [f32; A],
    /// The network's value of the state this node was expanded from
    value: f32,
    visits: [u32; A],
    /// Sum over visits of (reward + discount * return after) for each action
    q_sum: [f32; A],
    children: [u32; A],
}

impl Node {
    fn new(legal: [bool; A], logits: [f32; A], value: f32) -> Self {
        Self { legal, logits, value, visits: [0; A], q_sum: [0.0; A], children: [NONE; A] }
    }

    fn total_visits(&self) -> u32 {
        self.visits.iter().sum()
    }

    fn prior(&self) -> [f32; A] {
        softmax(&self.logits, &self.legal)
    }

    fn q(&self, a: usize) -> f32 {
        self.q_sum[a] / self.visits[a] as f32
    }

    /// The mixed value estimate (mctx's _compute_mixed_value): the network's value mixed with
    /// the prior-weighted Q values of the visited actions.
    fn mixed_value(&self) -> f32 {
        let prior = self.prior();
        let total = self.total_visits() as f32;
        let (mut prior_visited, mut weighted_q) = (0.0f32, 0.0f32);
        for a in 0..A {
            if self.legal[a] && self.visits[a] > 0 {
                prior_visited += prior[a];
                weighted_q += prior[a] * self.q(a);
            }
        }
        if prior_visited <= 0.0 {
            return self.value;
        }
        (self.value + total * weighted_q / prior_visited) / (1.0 + total)
    }

    /// sigma(completed Q), with Q rescaled to [0, 1] over the legal actions; 0 for illegal ones.
    fn sigma_q(&self, config: &GumbelConfig) -> [f32; A] {
        let mixed = self.mixed_value();
        let mut completed = [0.0f32; A];
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for a in 0..A {
            if self.legal[a] {
                completed[a] = if self.visits[a] > 0 { self.q(a) } else { mixed };
                lo = lo.min(completed[a]);
                hi = hi.max(completed[a]);
            }
        }
        let max_visits = *self.visits.iter().max().unwrap() as f32;
        let scale = (config.c_visit + max_visits) * config.c_scale;
        let mut sigma = [0.0f32; A];
        for a in 0..A {
            if self.legal[a] {
                let normalized = (completed[a] - lo) / (hi - lo).max(1e-8);
                sigma[a] = scale * normalized;
            }
        }
        sigma
    }

    /// The improved policy softmax(logits + sigma(completed Q)).
    fn improved_policy(&self, config: &GumbelConfig) -> [f32; A] {
        let sigma = self.sigma_q(config);
        let mut scores = [0.0f32; A];
        for a in 0..A {
            scores[a] = self.logits[a] + sigma[a];
        }
        softmax(&scores, &self.legal)
    }

    /// The interior selection rule: argmax(pi'(a) - N(a) / (1 + sum N)).
    fn select_interior(&self, config: &GumbelConfig) -> usize {
        let policy = self.improved_policy(config);
        let total = self.total_visits() as f32;
        argmax_legal(&self.legal, |a| policy[a] - self.visits[a] as f32 / (1.0 + total))
    }

    /// PUCT's choice among the legal actions, with `prior` (the root's may include noise) and Q
    /// rescaled by the tree's min and max Q.
    fn select_puct(&self, prior: &[f32; A], bounds: (f32, f32), config: &GumbelConfig) -> usize {
        let (lo, hi) = bounds;
        let normalize = |q: f32| if hi > lo { (q - lo) / (hi - lo) } else { 0.0 };
        let unvisited = normalize(self.mixed_value());
        let n = self.total_visits() as f32 + 1.0;
        let c = config.c_puct + ((n + config.c_puct_base + 1.0) / config.c_puct_base).ln();
        argmax_legal(&self.legal, |a| {
            let q = if self.visits[a] > 0 { normalize(self.q(a)) } else { unvisited };
            q + c * prior[a] * n.sqrt() / (1.0 + self.visits[a] as f32)
        })
    }

    /// The visit distribution over the legal actions (PUCT's policy target).
    fn visit_policy(&self) -> [f32; A] {
        let total = self.total_visits() as f32;
        if total == 0.0 {
            return self.prior();
        }
        std::array::from_fn(|a| if self.legal[a] { self.visits[a] as f32 / total } else { 0.0 })
    }

    /// The node's search value: the network's value and the visits' returns, averaged.
    fn search_value(&self) -> f32 {
        let total = self.total_visits() as f32;
        (self.value + self.q_sum.iter().sum::<f32>()) / (1.0 + total)
    }
}

fn softmax(scores: &[f32; A], legal: &[bool; A]) -> [f32; A] {
    let max = (0..A).filter(|&a| legal[a]).map(|a| scores[a]).fold(f32::NEG_INFINITY, f32::max);
    let mut out = [0.0f32; A];
    let mut sum = 0.0;
    for a in 0..A {
        if legal[a] {
            out[a] = (scores[a] - max).exp();
            sum += out[a];
        }
    }
    for p in out.iter_mut() {
        *p /= sum;
    }
    out
}

fn argmax_legal(legal: &[bool; A], score: impl Fn(usize) -> f32) -> usize {
    let mut best = usize::MAX;
    let mut best_score = f32::NEG_INFINITY;
    for a in 0..A {
        if legal[a] {
            let s = score(a);
            if best == usize::MAX || s > best_score {
                best = a;
                best_score = s;
            }
        }
    }
    assert!(best != usize::MAX, "no legal actions");
    best
}

/// mctx's get_sequence_of_considered_visits: the visit count the root action of each
/// simulation must have, which makes the root do sequential halving.
fn considered_visit_sequence(num_considered: usize, num_simulations: usize) -> Vec<u32> {
    if num_considered <= 1 {
        return (0..num_simulations as u32).collect();
    }
    let log2max = (num_considered as f32).log2().ceil() as usize;
    let mut sequence = Vec::with_capacity(num_simulations);
    let mut visits = vec![0u32; num_considered];
    let mut considered = num_considered;
    while sequence.len() < num_simulations {
        let extra = (num_simulations / (log2max * considered)).max(1);
        for _ in 0..extra {
            sequence.extend_from_slice(&visits[..considered]);
            for v in visits[..considered].iter_mut() {
                *v += 1;
            }
        }
        considered = (considered / 2).max(2);
    }
    sequence.truncate(num_simulations);
    sequence
}

/// Where the search's per-game work runs (GumbelConfig::threads).
enum Exec {
    Here,
    Global,
    Pool(rayon::ThreadPool),
}

impl Exec {
    fn new(threads: usize) -> PyResult<Self> {
        Ok(match threads {
            1 => Exec::Here,
            0 => Exec::Global,
            n => Exec::Pool(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n)
                    .build()
                    .map_err(|e| PyValueError::new_err(format!("couldn't start {n} search threads: {e}")))?,
            ),
        })
    }

    fn each<T: Send>(&self, items: &mut [T], f: impl Fn(&mut T) + Sync + Send) {
        match self {
            Exec::Here => items.iter_mut().for_each(f),
            Exec::Global => items.par_iter_mut().for_each(f),
            Exec::Pool(pool) => pool.install(|| items.par_iter_mut().for_each(f)),
        }
    }
}

/// One game's search: its tree, root noise and pending simulation.
struct Search {
    root_env: PacmanGym,
    nodes: Vec<Node>,
    gumbel: [f32; A],
    sequence: Vec<u32>,
    rng: SmallRng,
    /// PUCT: the root's prior, with exploration noise, and the min and max Q seen in the tree
    root_prior: [f32; A],
    q_bounds: (f32, f32),
    /// The pending simulation: (node, action, scaled reward) per move, and the state reached
    path: Vec<(u32, usize, f32)>,
    leaf: Leaf,
    /// The network's (value, logits) for an expanded leaf, once evaluated
    evaluation: Option<(f32, [f32; A])>,
}

enum Leaf {
    /// The game ended: nothing to evaluate
    Terminal,
    /// A state no node has been made for yet, to evaluate and add
    Expand(PacmanGym),
    /// The depth limit: bootstrap from this existing node's network value
    Bootstrap(u32),
}

impl Search {
    /// The root's score for considered actions: Gumbel noise + logits + sigma(completed Q).
    fn root_scores(&self, config: &GumbelConfig) -> [f32; A] {
        let root = &self.nodes[0];
        let sigma = root.sigma_q(config);
        let mut scores = [f32::NEG_INFINITY; A];
        for a in 0..A {
            if root.legal[a] {
                scores[a] = self.gumbel[a] + root.logits[a] + sigma[a];
            }
        }
        scores
    }

    /// The root action of simulation `k` (sequential halving; PUCT with the root's prior).
    fn root_action(&self, k: usize, config: &GumbelConfig) -> usize {
        let root = &self.nodes[0];
        if config.puct {
            return root.select_puct(&self.root_prior, self.q_bounds, config);
        }
        let target = self.sequence[k];
        let scores = self.root_scores(config);
        let eligible: [bool; A] = std::array::from_fn(|a| root.legal[a] && root.visits[a] == target);
        if eligible.iter().any(|&e| e) {
            argmax_legal(&eligible, |a| scores[a])
        } else {
            argmax_legal(&root.legal, |a| scores[a])
        }
    }

    /// The action to play: the best-scoring of the most visited root actions (PUCT: sampled by
    /// visits when exploring, else the most visited, ties to the higher Q).
    fn chosen_action(&mut self, config: &GumbelConfig) -> usize {
        let root = &self.nodes[0];
        if config.puct {
            let max_visits = (0..A).filter(|&a| root.legal[a]).map(|a| root.visits[a]).max().unwrap();
            if config.gumbel_scale > 0.0 && max_visits > 0 {
                let weights: [f64; A] = std::array::from_fn(|a| {
                    if root.legal[a] && root.visits[a] > 0 {
                        (root.visits[a] as f64 / max_visits as f64).powf(1.0 / config.temperature as f64)
                    } else {
                        0.0
                    }
                });
                let total: f64 = weights.iter().sum();
                let mut u = self.rng.gen::<f64>() * total;
                for a in 0..A {
                    if weights[a] > 0.0 {
                        if u < weights[a] {
                            return a;
                        }
                        u -= weights[a];
                    }
                }
                return argmax_legal(&root.legal, |a| weights[a] as f32);
            }
            let eligible: [bool; A] = std::array::from_fn(|a| root.legal[a] && root.visits[a] == max_visits);
            return argmax_legal(&eligible, |a| if root.visits[a] > 0 { root.q(a) } else { 0.0 });
        }
        let max_visits = (0..A).filter(|&a| root.legal[a]).map(|a| root.visits[a]).max().unwrap();
        let scores = self.root_scores(config);
        let eligible: [bool; A] = std::array::from_fn(|a| root.legal[a] && root.visits[a] == max_visits);
        argmax_legal(&eligible, |a| scores[a])
    }

    /// Plays simulation `k` down the tree, leaving it pending in self.path / self.leaf.
    fn select(&mut self, k: usize, config: &GumbelConfig) {
        let mut env = self.root_env.clone();
        env.game_state.seed = self.rng.gen();
        if env.latency_max > 0 {
            // Its own delays, too; drawn only with latency on, so searches without it are unchanged
            env.latency_seed = self.rng.gen();
        }
        self.path.clear();
        let mut node = 0u32;
        let mut action = self.root_action(k, config);
        loop {
            let (reward, done) = env.step(Action::from_index(action));
            self.path.push((node, action, reward as f32 * config.reward_scale));
            if done {
                self.leaf = Leaf::Terminal;
                return;
            }
            let child = self.nodes[node as usize].children[action];
            if child == NONE {
                self.leaf = Leaf::Expand(env);
                return;
            }
            if self.path.len() >= config.max_depth {
                self.leaf = Leaf::Bootstrap(child);
                return;
            }
            node = child;
            action = if config.puct {
                let n = &self.nodes[node as usize];
                n.select_puct(&n.prior(), self.q_bounds, config)
            } else {
                self.nodes[node as usize].select_interior(config)
            };
            if !env.action_mask()[action] {
                // An open-loop node can be reached in a state where its chosen action is
                // illegal only if legality differs between random outcomes, which it doesn't
                // (it depends on Pacman's square alone); guard anyway
                action = argmax_legal(&env.action_mask(), |_| 0.0);
            }
        }
    }

    /// Backs the pending simulation up, adding a node for an expanded leaf.
    fn backprop(&mut self, config: &GumbelConfig) {
        let mut ret = match (&self.leaf, self.evaluation.take()) {
            (Leaf::Terminal, _) => 0.0,
            (Leaf::Bootstrap(node), _) => self.nodes[*node as usize].value,
            (Leaf::Expand(env), Some((value, logits))) => {
                let index = self.nodes.len() as u32;
                self.nodes.push(Node::new(env.action_mask(), logits, value));
                let (parent, action, _) = *self.path.last().unwrap();
                self.nodes[parent as usize].children[action] = index;
                value
            }
            (Leaf::Expand(_), None) => panic!("an expanded leaf needs an evaluation"),
        };
        for &(node, action, reward) in self.path.iter().rev() {
            ret = reward + config.discount * ret;
            let node = &mut self.nodes[node as usize];
            node.visits[action] += 1;
            node.q_sum[action] += ret;
            let q = node.q(action);
            self.q_bounds = (self.q_bounds.0.min(q), self.q_bounds.1.max(q));
        }
        self.leaf = Leaf::Terminal;
    }
}

type Evaluation = Vec<(f32, [f32; A])>;

fn evaluate(
    py: Python<'_>,
    evaluator: &Py<PyAny>,
    envs: &[&PacmanGym],
    obs_config: &PacmanGymConfiguration,
    exec: &Exec,
) -> PyResult<Evaluation> {
    let n = envs.len();
    let (c, h, w) = OBS_SHAPE;
    let mut obs = Array4::<f32>::zeros((n, c, h, w));
    let mut legal = Array2::<bool>::from_elem((n, A), false);
    py.detach(|| {
        let flat = obs.as_slice_mut().expect("a new array is contiguous");
        let mut jobs: Vec<(&mut [f32], &PacmanGym)> = flat.chunks_mut(c * h * w).zip(envs.iter().copied()).collect();
        exec.each(&mut jobs, |(out, env)| {
            let one = env.obs(obs_config);
            out.copy_from_slice(one.as_slice().expect("obs() is contiguous"));
        });
    });
    for (i, env) in envs.iter().enumerate() {
        for (a, ok) in env.action_mask().iter().enumerate() {
            legal[(i, a)] = *ok;
        }
    }
    let result = evaluator.call1(py, (obs.into_pyarray(py), legal.into_pyarray(py)))?;
    let (values, logits): (PyReadonlyArray1<f32>, PyReadonlyArray2<f32>) = result.extract(py)?;
    let (values, logits) = (values.as_array(), logits.as_array());
    if values.len() != n || logits.shape() != [n, A] {
        return Err(PyValueError::new_err(format!(
            "the evaluator returned values of shape {:?} and logits of shape {:?} for {n} states",
            values.shape(),
            logits.shape()
        )));
    }
    Ok((0..n).map(|i| (values[i], std::array::from_fn(|a| logits[(i, a)]))).collect())
}

/// Searches every game in `envs` (left unchanged) with `config.num_simulations` simulations.
///
/// Returns (actions (B,) uint8: the action to play; policy (B, 5) float32: the improved policy
/// (PUCT: the visit distribution), the policy target; value (B,) float32: the root's search value, in the network's units;
/// network_value (B,) float32: the network's value of the root; q (B, 5) float32: the root's
/// completed Q values, 0 for illegal actions).
#[pyfunction]
#[pyo3(signature = (envs, evaluator, config, obs_config, seed))]
#[allow(clippy::type_complexity)]
pub fn gumbel_search<'py>(
    py: Python<'py>,
    envs: Vec<PacmanGym>,
    evaluator: Py<PyAny>,
    config: GumbelConfig,
    obs_config: PacmanGymConfiguration,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<f32>>,
)> {
    if envs.iter().any(|env| env.is_done()) {
        return Err(PyValueError::new_err("gumbel_search was given a game that has ended"));
    }
    let n = envs.len();
    let exec = Exec::new(config.threads)?;
    // Evaluate the roots
    let roots = evaluate(py, &evaluator, &envs.iter().collect::<Vec<_>>(), &obs_config, &exec)?;
    let mut searches: Vec<Search> = envs
        .into_iter()
        .zip(roots)
        .enumerate()
        .map(|(i, (env, (value, logits)))| {
            let legal = env.action_mask();
            let mut rng = SmallRng::seed_from_u64(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let gumbel: [f32; A] = std::array::from_fn(|_| {
                let u: f32 = rng.gen_range(f32::EPSILON..1.0);
                -config.gumbel_scale * (-u.ln()).ln()
            });
            let num_legal = legal.iter().filter(|&&l| l).count();
            let mut nodes = Vec::with_capacity(config.num_simulations + 1);
            nodes.push(Node::new(legal, logits, value));
            let mut root_prior = nodes[0].prior();
            if config.puct && config.gumbel_scale > 0.0 && config.dirichlet_fraction > 0.0 && num_legal > 1 {
                let noise = Dirichlet::new_with_size(config.dirichlet_alpha / num_legal as f32, num_legal)
                    .expect("dirichlet_alpha is above 0 and there are at least 2 legal actions")
                    .sample(&mut rng);
                let mut noise = noise.into_iter();
                for a in 0..A {
                    if legal[a] {
                        let eps = config.dirichlet_fraction;
                        root_prior[a] = (1.0 - eps) * root_prior[a] + eps * noise.next().unwrap();
                    }
                }
            }
            Search {
                root_env: env,
                nodes,
                gumbel,
                sequence: considered_visit_sequence(num_legal, config.num_simulations),
                rng,
                root_prior,
                q_bounds: (f32::INFINITY, f32::NEG_INFINITY),
                path: Vec::new(),
                leaf: Leaf::Terminal,
                evaluation: None,
            }
        })
        .collect();

    for k in 0..config.num_simulations {
        py.detach(|| exec.each(&mut searches, |s| s.select(k, &config)));
        let to_expand: Vec<usize> =
            (0..n).filter(|&i| matches!(searches[i].leaf, Leaf::Expand(_))).collect();
        let leaf_envs: Vec<&PacmanGym> = to_expand
            .iter()
            .map(|&i| match &searches[i].leaf {
                Leaf::Expand(env) => env,
                _ => unreachable!(),
            })
            .collect();
        if !leaf_envs.is_empty() {
            let evaluations = evaluate(py, &evaluator, &leaf_envs, &obs_config, &exec)?;
            for (&i, e) in to_expand.iter().zip(evaluations) {
                searches[i].evaluation = Some(e);
            }
        }
        py.detach(|| exec.each(&mut searches, |s| s.backprop(&config)));
    }

    let mut actions = Array1::<u8>::zeros(n);
    let mut policy = Array2::<f32>::zeros((n, A));
    let mut value = Array1::<f32>::zeros(n);
    let mut network_value = Array1::<f32>::zeros(n);
    let mut q = Array2::<f32>::zeros((n, A));
    for (i, s) in searches.iter_mut().enumerate() {
        actions[i] = s.chosen_action(&config) as u8;
        let root = &s.nodes[0];
        let improved = if config.puct { root.visit_policy() } else { root.improved_policy(&config) };
        let mixed = root.mixed_value();
        for a in 0..A {
            policy[(i, a)] = improved[a];
            if root.legal[a] {
                q[(i, a)] = if root.visits[a] > 0 { root.q(a) } else { mixed };
            }
        }
        value[i] = root.search_value();
        network_value[i] = root.value;
    }
    Ok((
        actions.into_pyarray(py),
        policy.into_pyarray(py),
        value.into_pyarray(py),
        network_value.into_pyarray(py),
        q.into_pyarray(py),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequential_halving_visits() {
        // 4 actions, 16 simulations: 2 rounds of all 4, then 4 rounds of the best 2 (mctx)
        let seq = considered_visit_sequence(4, 16);
        assert_eq!(seq, vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]);
        assert_eq!(considered_visit_sequence(1, 3), vec![0, 1, 2]);
        assert_eq!(considered_visit_sequence(5, 7).len(), 7);
    }

    fn puct_config() -> GumbelConfig {
        GumbelConfig::new(8, 0.99, 0.02, 50.0, 0.1, 0.0, 64, 1, true, 1.25, 19652.0, 10.0, 0.25, 1.0).unwrap()
    }

    #[test]
    fn puct_starts_with_the_prior_then_follows_q() {
        let config = puct_config();
        let legal = [true, true, true, false, false];
        let mut node = Node::new(legal, [0.0, 2.0, 0.0, 5.0, 0.0], 0.0);
        // Nothing visited: the most likely legal action (the illegal one's logit doesn't count)
        assert_eq!(node.select_puct(&node.prior(), (f32::INFINITY, f32::NEG_INFINITY), &config), 1);
        // Once action 1 has looked bad many times and action 0 good, action 0 wins
        node.visits = [3, 20, 0, 0, 0];
        node.q_sum = [3.0 * 1.0, 20.0 * -1.0, 0.0, 0.0, 0.0];
        assert_eq!(node.select_puct(&node.prior(), (-1.0, 1.0), &config), 0);
    }

    #[test]
    fn puct_explores_less_visited_actions() {
        let config = puct_config();
        let legal = [true, true, false, false, false];
        let mut node = Node::new(legal, [0.0, 0.0, 0.0, 0.0, 0.0], 0.0);
        // Equal priors and equal Q: the less visited action gets the bigger bonus
        node.visits = [10, 2, 0, 0, 0];
        node.q_sum = [5.0, 1.0, 0.0, 0.0, 0.0];
        assert_eq!(node.select_puct(&node.prior(), (0.0, 1.0), &config), 1);
    }

    #[test]
    fn visit_policy_is_the_visit_share() {
        let mut node = Node::new([true, true, false, true, false], [0.0; A], 0.0);
        node.visits = [1, 3, 0, 0, 0];
        assert_eq!(node.visit_policy(), [0.25, 0.75, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn softmax_ignores_illegal_actions() {
        let p = softmax(&[1.0, 100.0, 1.0, 0.0, 0.0], &[true, false, true, false, false]);
        assert!((p[0] - 0.5).abs() < 1e-6 && (p[2] - 0.5).abs() < 1e-6);
        assert_eq!(p[1], 0.0);
    }
}
