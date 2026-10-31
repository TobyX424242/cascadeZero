use crate::eval::{terminal_value_for_player, PositionEvaluator};
use crate::error::DataCollectionError;
use crate::model::{Action, ActionStatistics};
use crate::rng::{splitmix64, SmallRng};
use crate::GameState;

#[derive(Debug, Clone, PartialEq)]
pub struct MctsConfig {
    pub simulations: u32,
    pub c_puct: f32,
}

impl MctsConfig {
    pub fn with_simulations(mut self, simulations: u32) -> Self {
        self.simulations = simulations.max(1);
        self
    }

    pub fn with_c_puct(mut self, c_puct: f32) -> Self {
        self.c_puct = c_puct.max(0.1);
        self
    }
}

impl Default for MctsConfig {
    fn default() -> Self {
        Self {
            simulations: 128,
            c_puct: 1.35,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RootDirichletConfig {
    pub alpha: f32,
    pub exploration_fraction: f32,
    pub seed: u64,
}

impl RootDirichletConfig {
    pub fn with_alpha(mut self, alpha: f32) -> Self {
        self.alpha = alpha.max(1e-3);
        self
    }

    pub fn with_exploration_fraction(mut self, exploration_fraction: f32) -> Self {
        self.exploration_fraction = exploration_fraction.clamp(0.0, 1.0);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for RootDirichletConfig {
    fn default() -> Self {
        Self {
            alpha: 0.30,
            exploration_fraction: 0.25,
            seed: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SearchOptions {
    pub root_dirichlet: Option<RootDirichletConfig>,
}

impl SearchOptions {
    pub fn with_root_dirichlet(mut self, root_dirichlet: RootDirichletConfig) -> Self {
        self.root_dirichlet = Some(root_dirichlet);
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub root_value: f32,
    pub action_statistics: Vec<ActionStatistics>,
    pub total_nodes: u32,
    pub leaf_evaluations: u32,
}

#[derive(Debug, Clone)]
struct Edge {
    action: Action,
    prior: f32,
    visits: u32,
    value_sum: f32,
    child: Option<usize>,
}

#[derive(Debug, Clone)]
struct Node {
    state: GameState,
    visits: u32,
    value_sum: f32,
    terminal: Option<crate::TerminalStatus>,
    expanded: bool,
    edges: Vec<Edge>,
}

impl Node {
    fn new(state: GameState) -> Self {
        let terminal = state.terminal_status();
        Self {
            state,
            visits: 0,
            value_sum: 0.0,
            terminal,
            expanded: false,
            edges: Vec::new(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mcts<E> {
    config: MctsConfig,
    evaluator: E,
}

impl<E> Mcts<E>
where
    E: PositionEvaluator,
{
    pub fn new(config: MctsConfig, evaluator: E) -> Self {
        Self { config, evaluator }
    }

    pub fn search(&self, root_state: &GameState) -> Result<SearchResult, DataCollectionError> {
        self.search_with_options(root_state, SearchOptions::default())
    }

    pub fn search_with_options(
        &self,
        root_state: &GameState,
        options: SearchOptions,
    ) -> Result<SearchResult, DataCollectionError> {
        let mut arena = vec![Node::new(root_state.clone())];
        let mut leaf_evaluations = 0;

        for _ in 0..self.config.simulations {
            self.simulate(0, &mut arena, &mut leaf_evaluations, &options)?;
        }

        let root = &arena[0];
        let root_visits: u32 = root.edges.iter().map(|edge| edge.visits).sum();
        let root_value = if root_visits > 0 {
            root.edges.iter().map(|edge| edge.value_sum).sum::<f32>() / root_visits as f32
        } else if root.visits > 0 {
            root.value_sum / root.visits as f32
        } else {
            0.0
        };

        let mut action_statistics: Vec<ActionStatistics> = root
            .edges
            .iter()
            .map(|edge| ActionStatistics::new(edge.action.clone(), edge.visits, edge.value_sum).with_prior(edge.prior))
            .collect();
        action_statistics.sort_by(|left, right| {
            right
                .visits
                .cmp(&left.visits)
                .then_with(|| right.value_mean.total_cmp(&left.value_mean))
                .then_with(|| left.action.encoded_index().cmp(&right.action.encoded_index()))
        });

        Ok(SearchResult {
            root_value,
            action_statistics,
            total_nodes: arena.len() as u32,
            leaf_evaluations,
        })
    }

    fn simulate(
        &self,
        node_index: usize,
        arena: &mut Vec<Node>,
        leaf_evaluations: &mut u32,
        options: &SearchOptions,
    ) -> Result<f32, DataCollectionError> {
        if let Some(terminal) = arena[node_index].terminal {
            let value = terminal_value_for_player(terminal, arena[node_index].state.turn());
            arena[node_index].visits += 1;
            arena[node_index].value_sum += value;
            return Ok(value);
        }

        if !arena[node_index].expanded {
            let evaluation = self.evaluator.evaluate(&arena[node_index].state)?;
            let mut priors = normalize_priors(evaluation.priors);
            if node_index == 0 {
                apply_root_dirichlet_noise(&mut priors, options.root_dirichlet);
            }

            arena[node_index].edges = priors
                .into_iter()
                .map(|(action, prior)| Edge {
                    action,
                    prior,
                    visits: 0,
                    value_sum: 0.0,
                    child: None,
                })
                .collect();
            arena[node_index].expanded = true;
            arena[node_index].visits += 1;
            arena[node_index].value_sum += evaluation.value;
            *leaf_evaluations += 1;
            return Ok(evaluation.value);
        }

        let selected_edge_index = select_edge(&arena[node_index], self.config.c_puct);
        let child_index = if let Some(child_index) = arena[node_index].edges[selected_edge_index].child {
            child_index
        } else {
            let child_state = arena[node_index]
                .state
                .next_state(arena[node_index].edges[selected_edge_index].action.clone())?;
            let child_index = arena.len();
            arena.push(Node::new(child_state));
            arena[node_index].edges[selected_edge_index].child = Some(child_index);
            child_index
        };

        let child_value = self.simulate(child_index, arena, leaf_evaluations, options)?;
        // Apply value discounting to strictly prefer shorter paths to win and longer paths to lose.
        // Without this, MCTS may "play with its food" and delay forced wins.
        let value = -child_value * 0.999;

        arena[node_index].visits += 1;
        arena[node_index].value_sum += value;

        let edge = &mut arena[node_index].edges[selected_edge_index];
        edge.visits += 1;
        edge.value_sum += value;

        Ok(value)
    }
}

fn normalize_priors(priors: Vec<(Action, f32)>) -> Vec<(Action, f32)> {
    if priors.is_empty() {
        return priors;
    }

    let positive_total: f32 = priors
        .iter()
        .map(|(_, prior)| prior.max(0.0))
        .sum();

    if positive_total <= 0.0 {
        let uniform = 1.0 / priors.len() as f32;
        return priors
            .into_iter()
            .map(|(action, _)| (action, uniform))
            .collect();
    }

    priors
        .into_iter()
        .map(|(action, prior)| (action, prior.max(0.0) / positive_total))
        .collect()
}

fn apply_root_dirichlet_noise(
    priors: &mut [(Action, f32)],
    config: Option<RootDirichletConfig>,
) {
    let Some(config) = config else {
        return;
    };

    if priors.len() <= 1 || config.alpha <= 0.0 || config.exploration_fraction <= 0.0 {
        return;
    }

    let mut rng = SmallRng::new(splitmix64(config.seed));
    let noise = rng.sample_dirichlet(config.alpha, priors.len());
    let mix = config.exploration_fraction.clamp(0.0, 1.0);

    for ((_, prior), noise_value) in priors.iter_mut().zip(noise.into_iter()) {
        *prior = (*prior * (1.0 - mix) + noise_value * mix).max(0.0);
    }

    let total: f32 = priors.iter().map(|(_, prior)| *prior).sum();
    if total <= 0.0 || !total.is_finite() {
        let uniform = 1.0 / priors.len() as f32;
        for (_, prior) in priors {
            *prior = uniform;
        }
        return;
    }

    for (_, prior) in priors {
        *prior /= total;
    }
}

fn select_edge(node: &Node, c_puct: f32) -> usize {
    let total_visits: u32 = node.edges.iter().map(|edge| edge.visits).sum();
    let sqrt_total = (total_visits.max(1) as f32).sqrt();

    let mut best_index = 0;
    let mut best_score = f32::NEG_INFINITY;

    // First Play Urgency (FPU) Reduction: penalize unexplored nodes.
    // If we simply use the parent's Q-value, MCTS searches too broadly when optimistic,
    // wasting simulations on breadth and missing deeper tactical blunders (like suicides).
    // Subtracting 0.1 forces MCTS to deepen the search tree on the best prior lines.
    let fpu = if node.visits > 0 {
        node.value_sum / node.visits as f32 - 0.1
    } else {
        -0.1
    };

    for (index, edge) in node.edges.iter().enumerate() {
        let q_value = if edge.visits > 0 {
            edge.value_sum / edge.visits as f32
        } else {
            fpu
        };
        let u_value = c_puct * edge.prior * sqrt_total / (1.0 + edge.visits as f32);
        let score = q_value + u_value;

        if score > best_score + 1e-6
            || ((score - best_score).abs() <= 1e-6
                && edge.action.encoded_index() < node.edges[best_index].action.encoded_index())
        {
            best_score = score;
            best_index = index;
        }
    }

    best_index
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::{Evaluation, PositionEvaluator};

    #[derive(Debug, Clone, Copy)]
    struct FixedEvaluator;

    impl PositionEvaluator for FixedEvaluator {
        fn evaluate(&self, state: &GameState) -> Result<Evaluation, DataCollectionError> {
            let priors = state
                .legal_actions()
                .into_iter()
                .enumerate()
                .map(|(index, action)| (action, if index == 0 { 1.0 } else { 0.0 }))
                .collect();
            Ok(Evaluation { value: 0.0, priors })
        }
    }

    #[test]
    fn root_dirichlet_noise_perturbs_root_priors() {
        let search = Mcts::new(MctsConfig::default().with_simulations(1), FixedEvaluator);
        let result = search
            .search_with_options(
                &GameState::new(),
                SearchOptions::default().with_root_dirichlet(
                    RootDirichletConfig::default().with_seed(20260421),
                ),
            )
            .unwrap();

        let priors: Vec<f32> = result
            .action_statistics
            .iter()
            .map(|statistics| statistics.prior.unwrap())
            .collect();
        let total: f32 = priors.iter().sum();

        assert!((total - 1.0).abs() < 1e-5);
        assert!(priors[0] < 1.0);
        assert!(priors.iter().skip(1).any(|prior| *prior > 0.0));
    }
}
