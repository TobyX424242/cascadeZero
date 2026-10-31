use crate::action_selection::{best_action, sample_search_action};
use crate::collector::{SearchIterationRecord, TrainingDataCollector};
use crate::engine::{DrawReason, GameState, TerminalStatus};
use crate::eval::PositionEvaluator;
use crate::error::DataCollectionError;
use crate::mcts::{Mcts, RootDirichletConfig, SearchOptions, SearchResult};
use crate::model::{Action, SearchMetadata};
use crate::progress::PeriodicReporter;
use crate::rng::{splitmix64, SmallRng};
use crate::sink::DatasetSink;
use rayon::prelude::*;
use serde_json::json;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PROGRESS_REPORT_INTERVAL_MILLIS: u64 = 1000;
const ROOT_DIRICHLET_ALPHA_MIN: f32 = 0.1;
const ROOT_DIRICHLET_ALPHA_MAX: f32 = 0.8;

#[derive(Debug, Clone, PartialEq)]
pub struct SelfPlayConfig {
    pub games: u32,
    pub sampling_temperature: f32,
    pub temperature_drop_ply: u16,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub progress_log_every_plies: u16,
    pub slow_search_log_millis: u64,
    pub log_game_summaries: bool,
    pub seed: u64,
}

impl SelfPlayConfig {
    pub fn with_games(mut self, games: u32) -> Self {
        self.games = games.max(1);
        self
    }

    pub fn with_sampling_temperature(mut self, sampling_temperature: f32) -> Self {
        self.sampling_temperature = sampling_temperature.max(0.05);
        self
    }

    pub fn with_temperature_drop_ply(mut self, temperature_drop_ply: u16) -> Self {
        self.temperature_drop_ply = temperature_drop_ply;
        self
    }

    pub fn with_root_dirichlet_alpha(mut self, root_dirichlet_alpha: f32) -> Self {
        self.root_dirichlet_alpha = root_dirichlet_alpha.max(0.0);
        self
    }

    pub fn with_root_exploration_fraction(mut self, root_exploration_fraction: f32) -> Self {
        self.root_exploration_fraction = root_exploration_fraction.clamp(0.0, 1.0);
        self
    }

    pub fn with_progress_log_every_plies(mut self, progress_log_every_plies: u16) -> Self {
        self.progress_log_every_plies = progress_log_every_plies;
        self
    }

    pub fn with_slow_search_log_millis(mut self, slow_search_log_millis: u64) -> Self {
        self.slow_search_log_millis = slow_search_log_millis;
        self
    }

    pub fn with_log_game_summaries(mut self, log_game_summaries: bool) -> Self {
        self.log_game_summaries = log_game_summaries;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for SelfPlayConfig {
    fn default() -> Self {
        Self {
            games: 1,
            sampling_temperature: 1.0,
            temperature_drop_ply: 12,
            root_dirichlet_alpha: 10.0,
            root_exploration_fraction: 0.25,
            progress_log_every_plies: 0,
            slow_search_log_millis: 0,
            log_game_summaries: false,
            seed: 0x5eed_ca5cade,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParallelSelfPlayConfig {
    pub threads: usize,
}

impl ParallelSelfPlayConfig {
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }
}

impl Default for ParallelSelfPlayConfig {
    fn default() -> Self {
        Self {
            threads: std::thread::available_parallelism()
                .map(|parallelism| parallelism.get())
                .unwrap_or(1),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SelfPlaySummary {
    pub games_played: u32,
    pub red_wins: u32,
    pub blue_wins: u32,
    pub draws: u32,
    pub samples_written: u64,
    pub total_plies: u64,
}

impl SelfPlaySummary {
    pub fn merge(&mut self, other: Self) {
        self.games_played += other.games_played;
        self.red_wins += other.red_wins;
        self.blue_wins += other.blue_wins;
        self.draws += other.draws;
        self.samples_written += other.samples_written;
        self.total_plies += other.total_plies;
    }
}

struct SelfPlayProgressState {
    started_at: Instant,
    total_games: u32,
    active_games: AtomicU32,
    completed_games: AtomicU32,
    red_wins: AtomicU32,
    blue_wins: AtomicU32,
    draws: AtomicU32,
    searched_plies: AtomicU64,
    samples_written: AtomicU64,
    total_search_millis: AtomicU64,
    max_search_millis: AtomicU64,
    slow_searches: AtomicU64,
    total_nodes: AtomicU64,
    total_leaf_evaluations: AtomicU64,
    generation: AtomicU64,
    last_reported_generation: AtomicU64,
}

impl SelfPlayProgressState {
    fn maybe_create(config: &SelfPlayConfig) -> Option<(Arc<Self>, PeriodicReporter)> {
        if !selfplay_progress_enabled(config) {
            return None;
        }

        let state = Arc::new(Self {
            started_at: Instant::now(),
            total_games: config.games,
            active_games: AtomicU32::new(0),
            completed_games: AtomicU32::new(0),
            red_wins: AtomicU32::new(0),
            blue_wins: AtomicU32::new(0),
            draws: AtomicU32::new(0),
            searched_plies: AtomicU64::new(0),
            samples_written: AtomicU64::new(0),
            total_search_millis: AtomicU64::new(0),
            max_search_millis: AtomicU64::new(0),
            slow_searches: AtomicU64::new(0),
            total_nodes: AtomicU64::new(0),
            total_leaf_evaluations: AtomicU64::new(0),
            generation: AtomicU64::new(0),
            last_reported_generation: AtomicU64::new(u64::MAX),
        });

        let snapshot_state = Arc::clone(&state);
        let reporter = PeriodicReporter::spawn(PROGRESS_REPORT_INTERVAL_MILLIS, move || {
            snapshot_state.snapshot_line()
        });
        Some((state, reporter))
    }

    fn game_started(&self) {
        self.active_games.fetch_add(1, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    fn record_search(
        &self,
        elapsed_millis: u64,
        total_nodes: u64,
        total_leaf_evaluations: u64,
        slow_search_log_millis: u64,
    ) {
        self.searched_plies.fetch_add(1, Ordering::Relaxed);
        self.total_search_millis
            .fetch_add(elapsed_millis, Ordering::Relaxed);
        update_atomic_max(&self.max_search_millis, elapsed_millis);
        if slow_search_log_millis > 0 && elapsed_millis >= slow_search_log_millis {
            self.slow_searches.fetch_add(1, Ordering::Relaxed);
        }
        self.total_nodes.fetch_add(total_nodes, Ordering::Relaxed);
        self.total_leaf_evaluations
            .fetch_add(total_leaf_evaluations, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    fn record_game_finished(&self, terminal: TerminalStatus) {
        self.active_games.fetch_sub(1, Ordering::Relaxed);
        self.completed_games.fetch_add(1, Ordering::Relaxed);
        match terminal {
            TerminalStatus::Win {
                winner: crate::Player::Red,
                ..
            } => {
                self.red_wins.fetch_add(1, Ordering::Relaxed);
            }
            TerminalStatus::Win {
                winner: crate::Player::Blue,
                ..
            } => {
                self.blue_wins.fetch_add(1, Ordering::Relaxed);
            }
            TerminalStatus::Draw(_) => {
                self.draws.fetch_add(1, Ordering::Relaxed);
            }
        }
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    fn record_samples_written(&self, samples_written: u64) {
        if samples_written == 0 {
            return;
        }
        self.samples_written
            .fetch_add(samples_written, Ordering::Relaxed);
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    fn snapshot_line(&self) -> Option<String> {
        let generation = self.generation.load(Ordering::Relaxed);
        let previous = self
            .last_reported_generation
            .swap(generation, Ordering::Relaxed);
        let active_games = self.active_games.load(Ordering::Relaxed);
        if generation == previous && active_games == 0 {
            return None;
        }

        let elapsed_ms = self.started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let completed_games = self.completed_games.load(Ordering::Relaxed);
        let red_wins = self.red_wins.load(Ordering::Relaxed);
        let blue_wins = self.blue_wins.load(Ordering::Relaxed);
        let draws = self.draws.load(Ordering::Relaxed);
        let searched_plies = self.searched_plies.load(Ordering::Relaxed);
        let samples_written = self.samples_written.load(Ordering::Relaxed);
        let total_search_millis = self.total_search_millis.load(Ordering::Relaxed);
        let max_search_millis = self.max_search_millis.load(Ordering::Relaxed);
        let slow_searches = self.slow_searches.load(Ordering::Relaxed);
        let total_nodes = self.total_nodes.load(Ordering::Relaxed);
        let total_leaf_evaluations = self.total_leaf_evaluations.load(Ordering::Relaxed);

        Some(format!(
            "progress kind=selfplay elapsed_ms={} games_completed={}/{} active_games={} red_wins={} blue_wins={} draws={} searched_plies={} samples_written={} avg_search_ms={:.2} max_search_ms={} slow_searches={} total_nodes={} total_leafs={} nodes_per_second={:.2}",
            elapsed_ms,
            completed_games,
            self.total_games,
            active_games,
            red_wins,
            blue_wins,
            draws,
            searched_plies,
            samples_written,
            average_millis(total_search_millis, searched_plies),
            max_search_millis,
            slow_searches,
            total_nodes,
            total_leaf_evaluations,
            nodes_per_second(total_nodes, total_search_millis),
        ))
    }
}

pub struct SelfPlayRunner<S, E>
where
    S: DatasetSink,
    E: PositionEvaluator,
{
    config: SelfPlayConfig,
    search: Mcts<E>,
    collector: TrainingDataCollector<S>,
    search_index: u64,
}

impl<S, E> SelfPlayRunner<S, E>
where
    S: DatasetSink,
    E: PositionEvaluator,
{
    pub fn new(
        config: SelfPlayConfig,
        search: Mcts<E>,
        collector: TrainingDataCollector<S>,
    ) -> Self {
        Self {
            config,
            search,
            collector,
            search_index: 0,
        }
    }

    pub fn play_games(&mut self) -> Result<SelfPlaySummary, DataCollectionError> {
        let mut summary = SelfPlaySummary::default();
        let progress = SelfPlayProgressState::maybe_create(&self.config);

        for game_index in 0..self.config.games {
            let game = play_single_game(
                &self.config,
                &self.search,
                game_index,
                SmallRng::new(splitmix64(self.config.seed ^ game_index as u64)),
                || {
                    let next_index = self.search_index;
                    self.search_index += 1;
                    next_index
                },
                progress.as_ref().map(|(state, _)| Arc::clone(state)),
                None,
            )?;
            let game_summary = write_completed_game(&mut self.collector, game)?;
            if let Some((state, _)) = &progress {
                state.record_samples_written(game_summary.samples_written);
            }
            summary.merge(game_summary);
        }

        self.collector.flush()?;
        drop(progress);
        Ok(summary)
    }

    pub fn into_collector(self) -> TrainingDataCollector<S> {
        self.collector
    }
}

pub fn run_parallel_self_play<S, E>(
    config: SelfPlayConfig,
    parallel: ParallelSelfPlayConfig,
    search: Mcts<E>,
    collector: TrainingDataCollector<S>,
) -> Result<(SelfPlaySummary, TrainingDataCollector<S>), DataCollectionError>
where
    S: DatasetSink + Send,
    E: PositionEvaluator + Send + Sync,
{
    if parallel.threads <= 1 || config.games <= 1 {
        let mut runner = SelfPlayRunner::new(config, search, collector);
        let summary = runner.play_games()?;
        return Ok((summary, runner.into_collector()));
    }

    let shared_collector = Arc::new(Mutex::new(collector));
    let worker_collector = Arc::clone(&shared_collector);
    let progress = SelfPlayProgressState::maybe_create(&config);
    let search_index = Arc::new(AtomicU64::new(0));
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel.threads)
        .build()?;

    let summary = thread_pool.install(|| {
        (0..config.games)
            .into_par_iter()
            .try_fold(SelfPlaySummary::default, |mut acc, game_index| {
                let worker_id = rayon::current_thread_index().map(|i| i as u32);
                let game = play_single_game(
                    &config,
                    &search,
                    game_index,
                    SmallRng::new(splitmix64(
                        config.seed
                            ^ (game_index as u64 + 1)
                                .wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    )),
                    || search_index.fetch_add(1, Ordering::Relaxed),
                    progress.as_ref().map(|(state, _)| Arc::clone(state)),
                    worker_id,
                )?;

                let mut collector = worker_collector.lock().map_err(|_| {
                    DataCollectionError::ParallelExecution(
                        "collector mutex was poisoned during parallel self-play".to_string(),
                    )
                })?;
                let game_summary = write_completed_game(&mut *collector, game)?;
                if let Some((state, _)) = &progress {
                    state.record_samples_written(game_summary.samples_written);
                }
                acc.merge(game_summary);
                Ok::<SelfPlaySummary, DataCollectionError>(acc)
            })
            .try_reduce(SelfPlaySummary::default, |mut left, right| {
                left.merge(right);
                Ok::<SelfPlaySummary, DataCollectionError>(left)
            })
    })?;

    drop(worker_collector);

    let mut collector = Arc::try_unwrap(shared_collector)
        .map_err(|_| {
            DataCollectionError::ParallelExecution(
                "parallel collector is still shared after completion".to_string(),
            )
        })?
        .into_inner()
        .map_err(|_| {
            DataCollectionError::ParallelExecution(
                "collector mutex was poisoned when finalizing self-play".to_string(),
            )
        })?;
    collector.flush()?;
    drop(progress);

    Ok((summary, collector))
}

#[derive(Debug, Clone)]
struct CompletedGame {
    game_index: u32,
    total_plies: u64,
    terminal: TerminalStatus,
    pending_records: Vec<SearchIterationRecord>,
}

fn play_single_game<E, F>(
    config: &SelfPlayConfig,
    search: &Mcts<E>,
    game_index: u32,
    mut rng: SmallRng,
    mut next_search_index: F,
    progress: Option<Arc<SelfPlayProgressState>>,
    worker_id: Option<u32>,
) -> Result<CompletedGame, DataCollectionError>
where
    E: PositionEvaluator,
    F: FnMut() -> u64,
{
    let mut state = GameState::new();
    let mut pending_records = Vec::new();

    if let Some(progress) = &progress {
        progress.game_started();
    }

    loop {
        if let Some(terminal) = state.terminal_status() {
            if let Some(progress) = &progress {
                progress.record_game_finished(terminal);
            }
            return Ok(CompletedGame {
                game_index,
                total_plies: state.ply() as u64,
                terminal,
                pending_records,
            });
        }

        let legal_action_count = if config.root_dirichlet_alpha > 0.0
            && config.root_exploration_fraction > 0.0
        {
            state.legal_actions().len()
        } else {
            0
        };
        let root_dirichlet_alpha = dynamic_root_dirichlet_alpha(config, legal_action_count);

        let started_at = Instant::now();
        let search_result = search.search_with_options(
            &state,
            search_options_for_self_play(config, root_dirichlet_alpha, &mut rng),
        )?;
        let elapsed_millis = started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;

        if let Some(progress) = &progress {
            progress.record_search(
                elapsed_millis,
                search_result.total_nodes as u64,
                search_result.leaf_evaluations as u64,
                config.slow_search_log_millis,
            );
        }

        if search_result.action_statistics.is_empty() {
            if let Some(progress) = &progress {
                progress.record_game_finished(TerminalStatus::Draw(DrawReason::Stalemate));
            }
            return Ok(CompletedGame {
                game_index,
                total_plies: state.ply() as u64,
                terminal: TerminalStatus::Draw(DrawReason::Stalemate),
                pending_records,
            });
        }

        let selected_action = select_action(config, &search_result, state.ply(), &mut rng);
        let mut search_metadata = SearchMetadata::new(next_search_index())
            .with_total_nodes(search_result.total_nodes)
            .with_leaf_evaluations(search_result.leaf_evaluations)
            .with_elapsed_millis(elapsed_millis);
        if let Some(root_dirichlet_alpha) = root_dirichlet_alpha {
            search_metadata = search_metadata
                .with_metadata("root_dirichlet_alpha", json!(root_dirichlet_alpha))
                .with_metadata(
                    "root_dirichlet_alpha_numerator",
                    json!(config.root_dirichlet_alpha),
                )
                .with_metadata("root_dirichlet_legal_actions", json!(legal_action_count))
                .with_metadata(
                    "root_exploration_fraction",
                    json!(config.root_exploration_fraction),
                );
        }
        if let Some(worker_id) = worker_id {
            search_metadata = search_metadata.with_worker_id(worker_id);
        }

        let record = SearchIterationRecord::new(state.to_snapshot(), search_metadata)
            .with_action_statistics(search_result.action_statistics.clone())
            .with_selected_action(selected_action.clone())
            .with_root_value(search_result.root_value)
            .with_metadata("game_index", json!(game_index));
        pending_records.push(record);

        state.apply_action(selected_action)?;
    }
}

fn write_completed_game<S: DatasetSink>(
    collector: &mut TrainingDataCollector<S>,
    completed_game: CompletedGame,
) -> Result<SelfPlaySummary, DataCollectionError> {
    let terminal_label = terminal_label(completed_game.terminal);
    let mut samples_written = 0u64;

    for record in completed_game.pending_records {
        let outcome = terminal_value_for_turn(completed_game.terminal, record.root_state.turn);
        collector.record(
            record
                .with_outcome_target(outcome)
                .with_metadata("game_index", json!(completed_game.game_index))
                .with_metadata("terminal", json!(terminal_label)),
        )?;
        samples_written += 1;
    }

    Ok(match completed_game.terminal {
        TerminalStatus::Win {
            winner: crate::Player::Red,
            ..
        } => SelfPlaySummary {
            games_played: 1,
            red_wins: 1,
            blue_wins: 0,
            draws: 0,
            samples_written,
            total_plies: completed_game.total_plies,
        },
        TerminalStatus::Win {
            winner: crate::Player::Blue,
            ..
        } => SelfPlaySummary {
            games_played: 1,
            red_wins: 0,
            blue_wins: 1,
            draws: 0,
            samples_written,
            total_plies: completed_game.total_plies,
        },
        TerminalStatus::Draw(_) => SelfPlaySummary {
            games_played: 1,
            red_wins: 0,
            blue_wins: 0,
            draws: 1,
            samples_written,
            total_plies: completed_game.total_plies,
        },
    })
}

fn select_action(
    config: &SelfPlayConfig,
    search_result: &SearchResult,
    ply: u16,
    rng: &mut SmallRng,
) -> Action {
    if ply < config.temperature_drop_ply {
        return sample_search_action(
            &search_result.action_statistics,
            config.sampling_temperature,
            rng,
        );
    }

    best_action(&search_result.action_statistics).action.clone()
}

fn dynamic_root_dirichlet_alpha(
    config: &SelfPlayConfig,
    legal_action_count: usize,
) -> Option<f32> {
    if config.root_dirichlet_alpha <= 0.0
        || config.root_exploration_fraction <= 0.0
        || legal_action_count <= 1
    {
        return None;
    }

    Some(
        (config.root_dirichlet_alpha / legal_action_count as f32)
            .clamp(ROOT_DIRICHLET_ALPHA_MIN, ROOT_DIRICHLET_ALPHA_MAX),
    )
}

fn search_options_for_self_play(
    config: &SelfPlayConfig,
    root_dirichlet_alpha: Option<f32>,
    rng: &mut SmallRng,
) -> SearchOptions {
    let Some(root_dirichlet_alpha) = root_dirichlet_alpha else {
        return SearchOptions::default();
    };

    SearchOptions::default().with_root_dirichlet(
        RootDirichletConfig::default()
            .with_alpha(root_dirichlet_alpha)
            .with_exploration_fraction(config.root_exploration_fraction)
            .with_seed(rng.next_u64()),
    )
}

fn terminal_value_for_turn(terminal: TerminalStatus, turn: crate::Player) -> f32 {
    match terminal {
        TerminalStatus::Win { winner, .. } if winner == turn => 1.0,
        TerminalStatus::Win { .. } => -1.0,
        TerminalStatus::Draw(_) => -0.0,
    }
}

fn terminal_label(terminal: TerminalStatus) -> &'static str {
    match terminal {
        TerminalStatus::Win {
            winner: crate::Player::Red,
            ..
        } => "red_win",
        TerminalStatus::Win {
            winner: crate::Player::Blue,
            ..
        } => "blue_win",
        TerminalStatus::Draw(DrawReason::ThreefoldRepetition) => "draw_threefold",
        TerminalStatus::Draw(DrawReason::Stalemate) => "draw_stalemate",
        TerminalStatus::Draw(DrawReason::TurnLimit) => "draw_turn_limit",
    }
}

fn selfplay_progress_enabled(config: &SelfPlayConfig) -> bool {
    config.progress_log_every_plies > 0
        || config.slow_search_log_millis > 0
        || config.log_game_summaries
}

fn average_millis(total_millis: u64, count: u64) -> f32 {
    if count == 0 {
        0.0
    } else {
        total_millis as f32 / count as f32
    }
}

fn nodes_per_second(total_nodes: u64, total_millis: u64) -> f32 {
    if total_millis == 0 {
        0.0
    } else {
        total_nodes as f32 / (total_millis as f32 / 1000.0)
    }
}

fn update_atomic_max(target: &AtomicU64, value: u64) {
    let mut current = target.load(Ordering::Relaxed);
    while value > current {
        match target.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => current = actual,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_root_dirichlet_alpha_scales_with_legal_action_count() {
        let config = SelfPlayConfig::default().with_root_dirichlet_alpha(10.0);

        assert_eq!(dynamic_root_dirichlet_alpha(&config, 0), None);
        assert_eq!(dynamic_root_dirichlet_alpha(&config, 1), None);

        let alpha = dynamic_root_dirichlet_alpha(&config, 20).expect("alpha should be enabled");
        assert!((alpha - 0.5).abs() < f32::EPSILON);

        let low_alpha =
            dynamic_root_dirichlet_alpha(&config, 200).expect("alpha should be clamped");
        assert!((low_alpha - ROOT_DIRICHLET_ALPHA_MIN).abs() < f32::EPSILON);

        let high_alpha =
            dynamic_root_dirichlet_alpha(&config, 2).expect("alpha should be clamped");
        assert!((high_alpha - ROOT_DIRICHLET_ALPHA_MAX).abs() < f32::EPSILON);
    }
}
