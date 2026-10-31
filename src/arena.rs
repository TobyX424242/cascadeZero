use crate::action_selection::{best_action, sample_search_action};
use crate::engine::{DrawReason, GameState, TerminalStatus};
use crate::eval::PositionEvaluator;
use crate::error::DataCollectionError;
use crate::mcts::{Mcts, SearchResult};
use crate::model::{Action, Player};
use crate::progress::PeriodicReporter;
use crate::rng::{splitmix64, SmallRng};
use rayon::prelude::*;
use serde::Serialize;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

const PROGRESS_REPORT_INTERVAL_MILLIS: u64 = 1000;

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArenaConfig {
    pub games: u32,
    pub seed: u64,
    pub opening_temperature_plies: u16,
    pub opening_temperature: f32,
    pub progress_log_every_plies: u16,
    pub slow_search_log_millis: u64,
    pub log_game_summaries: bool,
}

impl ArenaConfig {
    pub fn with_games(mut self, games: u32) -> Self {
        self.games = games.max(2);
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_opening_temperature_plies(mut self, opening_temperature_plies: u16) -> Self {
        self.opening_temperature_plies = opening_temperature_plies;
        self
    }

    pub fn with_opening_temperature(mut self, opening_temperature: f32) -> Self {
        self.opening_temperature = opening_temperature.max(0.05);
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
}

impl Default for ArenaConfig {
    fn default() -> Self {
        Self {
            games: 12,
            seed: 20260420,
            opening_temperature_plies: 8,
            opening_temperature: 1.0,
            progress_log_every_plies: 0,
            slow_search_log_millis: 0,
            log_game_summaries: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParallelArenaConfig {
    pub threads: usize,
}

impl ParallelArenaConfig {
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }
}

impl Default for ParallelArenaConfig {
    fn default() -> Self {
        Self {
            threads: std::thread::available_parallelism()
                .map(|parallelism| parallelism.get())
                .unwrap_or(1),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArenaOutcome {
    PlayerAWin,
    PlayerBWin,
    Draw,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ArenaSeat {
    Red,
    Blue,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ArenaGameResult {
    pub game_index: u32,
    pub opening_index: u32,
    pub player_a_seat: ArenaSeat,
    pub player_b_seat: ArenaSeat,
    pub outcome: ArenaOutcome,
    pub winner: Option<String>,
    pub terminal: String,
    pub total_plies: u64,
    pub player_a_moves: u64,
    pub player_b_moves: u64,
    pub player_a_search_millis: u64,
    pub player_b_search_millis: u64,
    pub player_a_total_nodes: u64,
    pub player_b_total_nodes: u64,
    pub player_a_leaf_evaluations: u64,
    pub player_b_leaf_evaluations: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct ArenaPlayerSummary {
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
    pub red_games: u32,
    pub blue_games: u32,
    pub red_wins: u32,
    pub blue_wins: u32,
    pub moves: u64,
    pub search_millis: u64,
    pub total_nodes: u64,
    pub leaf_evaluations: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ArenaSummary {
    pub config: ArenaConfig,
    pub player_a_name: String,
    pub player_b_name: String,
    pub games_played: u32,
    pub draws: u32,
    pub total_plies: u64,
    pub player_a: ArenaPlayerSummary,
    pub player_b: ArenaPlayerSummary,
    pub game_results: Vec<ArenaGameResult>,
}

impl ArenaSummary {
    fn new(config: ArenaConfig, player_a_name: String, player_b_name: String) -> Self {
        Self {
            config,
            player_a_name,
            player_b_name,
            games_played: 0,
            draws: 0,
            total_plies: 0,
            player_a: ArenaPlayerSummary::default(),
            player_b: ArenaPlayerSummary::default(),
            game_results: Vec::new(),
        }
    }

    pub fn player_a_score(&self) -> f32 {
        if self.games_played == 0 {
            return 0.0;
        }

        (self.player_a.wins as f32 + self.draws as f32 * 0.5) / self.games_played as f32
    }

    pub fn player_b_score(&self) -> f32 {
        if self.games_played == 0 {
            return 0.0;
        }

        (self.player_b.wins as f32 + self.draws as f32 * 0.5) / self.games_played as f32
    }

    pub fn player_a_avg_search_millis(&self) -> f32 {
        average_millis(self.player_a.search_millis, self.player_a.moves)
    }

    pub fn player_b_avg_search_millis(&self) -> f32 {
        average_millis(self.player_b.search_millis, self.player_b.moves)
    }

    pub fn player_a_nodes_per_second(&self) -> f32 {
        nodes_per_second(self.player_a.total_nodes, self.player_a.search_millis)
    }

    pub fn player_b_nodes_per_second(&self) -> f32 {
        nodes_per_second(self.player_b.total_nodes, self.player_b.search_millis)
    }

    fn record_game(&mut self, game: ArenaGameResult) {
        self.games_played += 1;
        self.total_plies += game.total_plies;

        match game.player_a_seat {
            ArenaSeat::Red => self.player_a.red_games += 1,
            ArenaSeat::Blue => self.player_a.blue_games += 1,
        }
        match game.player_b_seat {
            ArenaSeat::Red => self.player_b.red_games += 1,
            ArenaSeat::Blue => self.player_b.blue_games += 1,
        }

        self.player_a.moves += game.player_a_moves;
        self.player_b.moves += game.player_b_moves;
        self.player_a.search_millis += game.player_a_search_millis;
        self.player_b.search_millis += game.player_b_search_millis;
        self.player_a.total_nodes += game.player_a_total_nodes;
        self.player_b.total_nodes += game.player_b_total_nodes;
        self.player_a.leaf_evaluations += game.player_a_leaf_evaluations;
        self.player_b.leaf_evaluations += game.player_b_leaf_evaluations;

        match game.outcome {
            ArenaOutcome::PlayerAWin => {
                self.player_a.wins += 1;
                self.player_b.losses += 1;
                match game.player_a_seat {
                    ArenaSeat::Red => self.player_a.red_wins += 1,
                    ArenaSeat::Blue => self.player_a.blue_wins += 1,
                }
            }
            ArenaOutcome::PlayerBWin => {
                self.player_b.wins += 1;
                self.player_a.losses += 1;
                match game.player_b_seat {
                    ArenaSeat::Red => self.player_b.red_wins += 1,
                    ArenaSeat::Blue => self.player_b.blue_wins += 1,
                }
            }
            ArenaOutcome::Draw => {
                self.draws += 1;
                self.player_a.draws += 1;
                self.player_b.draws += 1;
            }
        }

        self.game_results.push(game);
    }
}

struct ArenaProgressState {
    started_at: Instant,
    total_games: u32,
    active_games: AtomicU32,
    completed_games: AtomicU32,
    draws: AtomicU32,
    total_plies: AtomicU64,
    slow_searches: AtomicU64,
    max_search_millis: AtomicU64,
    player_a_wins: AtomicU32,
    player_b_wins: AtomicU32,
    player_a_moves: AtomicU64,
    player_b_moves: AtomicU64,
    player_a_search_millis: AtomicU64,
    player_b_search_millis: AtomicU64,
    player_a_total_nodes: AtomicU64,
    player_b_total_nodes: AtomicU64,
    player_a_leaf_evaluations: AtomicU64,
    player_b_leaf_evaluations: AtomicU64,
    generation: AtomicU64,
    last_reported_generation: AtomicU64,
}

impl ArenaProgressState {
    fn maybe_create(config: &ArenaConfig) -> Option<(Arc<Self>, PeriodicReporter)> {
        if !arena_progress_enabled(config) {
            return None;
        }

        let state = Arc::new(Self {
            started_at: Instant::now(),
            total_games: config.games,
            active_games: AtomicU32::new(0),
            completed_games: AtomicU32::new(0),
            draws: AtomicU32::new(0),
            total_plies: AtomicU64::new(0),
            slow_searches: AtomicU64::new(0),
            max_search_millis: AtomicU64::new(0),
            player_a_wins: AtomicU32::new(0),
            player_b_wins: AtomicU32::new(0),
            player_a_moves: AtomicU64::new(0),
            player_b_moves: AtomicU64::new(0),
            player_a_search_millis: AtomicU64::new(0),
            player_b_search_millis: AtomicU64::new(0),
            player_a_total_nodes: AtomicU64::new(0),
            player_b_total_nodes: AtomicU64::new(0),
            player_a_leaf_evaluations: AtomicU64::new(0),
            player_b_leaf_evaluations: AtomicU64::new(0),
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
        current_player_is_a: bool,
        elapsed_millis: u64,
        total_nodes: u64,
        total_leaf_evaluations: u64,
        slow_search_log_millis: u64,
    ) {
        self.total_plies.fetch_add(1, Ordering::Relaxed);
        update_atomic_max(&self.max_search_millis, elapsed_millis);
        if slow_search_log_millis > 0 && elapsed_millis >= slow_search_log_millis {
            self.slow_searches.fetch_add(1, Ordering::Relaxed);
        }

        if current_player_is_a {
            self.player_a_moves.fetch_add(1, Ordering::Relaxed);
            self.player_a_search_millis
                .fetch_add(elapsed_millis, Ordering::Relaxed);
            self.player_a_total_nodes
                .fetch_add(total_nodes, Ordering::Relaxed);
            self.player_a_leaf_evaluations
                .fetch_add(total_leaf_evaluations, Ordering::Relaxed);
        } else {
            self.player_b_moves.fetch_add(1, Ordering::Relaxed);
            self.player_b_search_millis
                .fetch_add(elapsed_millis, Ordering::Relaxed);
            self.player_b_total_nodes
                .fetch_add(total_nodes, Ordering::Relaxed);
            self.player_b_leaf_evaluations
                .fetch_add(total_leaf_evaluations, Ordering::Relaxed);
        }

        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    fn record_game_finished(&self, outcome: &ArenaOutcome) {
        self.active_games.fetch_sub(1, Ordering::Relaxed);
        self.completed_games.fetch_add(1, Ordering::Relaxed);
        match outcome {
            ArenaOutcome::PlayerAWin => {
                self.player_a_wins.fetch_add(1, Ordering::Relaxed);
            }
            ArenaOutcome::PlayerBWin => {
                self.player_b_wins.fetch_add(1, Ordering::Relaxed);
            }
            ArenaOutcome::Draw => {
                self.draws.fetch_add(1, Ordering::Relaxed);
            }
        }
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
        let draws = self.draws.load(Ordering::Relaxed);
        let total_plies = self.total_plies.load(Ordering::Relaxed);
        let slow_searches = self.slow_searches.load(Ordering::Relaxed);
        let max_search_millis = self.max_search_millis.load(Ordering::Relaxed);
        let player_a_wins = self.player_a_wins.load(Ordering::Relaxed);
        let player_b_wins = self.player_b_wins.load(Ordering::Relaxed);
        let player_a_moves = self.player_a_moves.load(Ordering::Relaxed);
        let player_b_moves = self.player_b_moves.load(Ordering::Relaxed);
        let player_a_search_millis = self.player_a_search_millis.load(Ordering::Relaxed);
        let player_b_search_millis = self.player_b_search_millis.load(Ordering::Relaxed);
        let player_a_total_nodes = self.player_a_total_nodes.load(Ordering::Relaxed);
        let player_b_total_nodes = self.player_b_total_nodes.load(Ordering::Relaxed);
        let player_a_leaf_evaluations = self.player_a_leaf_evaluations.load(Ordering::Relaxed);
        let player_b_leaf_evaluations = self.player_b_leaf_evaluations.load(Ordering::Relaxed);
        let player_a_score = if completed_games == 0 {
            0.0
        } else {
            (player_a_wins as f32 + draws as f32 * 0.5) / completed_games as f32
        };
        let player_b_score = if completed_games == 0 {
            0.0
        } else {
            (player_b_wins as f32 + draws as f32 * 0.5) / completed_games as f32
        };

        Some(format!(
            "progress kind=arena elapsed_ms={} games_completed={}/{} active_games={} player_a_wins={} player_b_wins={} draws={} player_a_score={:.4} player_b_score={:.4} total_plies={} player_a_avg_ms={:.2} player_b_avg_ms={:.2} player_a_nps={:.2} player_b_nps={:.2} max_search_ms={} slow_searches={} player_a_leafs={} player_b_leafs={}",
            elapsed_ms,
            completed_games,
            self.total_games,
            active_games,
            player_a_wins,
            player_b_wins,
            draws,
            player_a_score,
            player_b_score,
            total_plies,
            average_millis(player_a_search_millis, player_a_moves),
            average_millis(player_b_search_millis, player_b_moves),
            nodes_per_second(player_a_total_nodes, player_a_search_millis),
            nodes_per_second(player_b_total_nodes, player_b_search_millis),
            max_search_millis,
            slow_searches,
            player_a_leaf_evaluations,
            player_b_leaf_evaluations,
        ))
    }
}

pub fn run_arena<E1, E2>(
    config: ArenaConfig,
    player_a_name: impl Into<String>,
    player_a_search: &Mcts<E1>,
    player_b_name: impl Into<String>,
    player_b_search: &Mcts<E2>,
) -> Result<ArenaSummary, DataCollectionError>
where
    E1: PositionEvaluator,
    E2: PositionEvaluator,
{
    let mut summary = ArenaSummary::new(config.clone(), player_a_name.into(), player_b_name.into());
    let progress = ArenaProgressState::maybe_create(&config);

    for game_index in 0..config.games {
        let opening_index = game_index / 2;
        let player_a_is_red = game_index % 2 == 0;
        let seed = splitmix64(config.seed ^ opening_index as u64);
        let game = play_arena_game(
            &config,
            game_index,
            opening_index,
            player_a_is_red,
            SmallRng::new(seed),
            player_a_search,
            player_b_search,
            &summary.player_a_name,
            &summary.player_b_name,
            progress.as_ref().map(|(state, _)| Arc::clone(state)),
        )?;
        summary.record_game(game);
    }

    drop(progress);

    Ok(summary)
}

pub fn run_parallel_arena<E1, E2>(
    config: ArenaConfig,
    parallel: ParallelArenaConfig,
    player_a_name: impl Into<String>,
    player_a_search: &Mcts<E1>,
    player_b_name: impl Into<String>,
    player_b_search: &Mcts<E2>,
) -> Result<ArenaSummary, DataCollectionError>
where
    E1: PositionEvaluator + Send + Sync,
    E2: PositionEvaluator + Send + Sync,
{
    if parallel.threads <= 1 || config.games <= 1 {
        return run_arena(
            config,
            player_a_name,
            player_a_search,
            player_b_name,
            player_b_search,
        );
    }

    let player_a_name = player_a_name.into();
    let player_b_name = player_b_name.into();
    let progress = ArenaProgressState::maybe_create(&config);
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel.threads)
        .build()?;

    let mut games = thread_pool.install(|| {
        (0..config.games)
            .into_par_iter()
            .try_fold(Vec::new, |mut acc, game_index| {
                let opening_index = game_index / 2;
                let player_a_is_red = game_index % 2 == 0;
                let seed = splitmix64(config.seed ^ opening_index as u64);
                let game = play_arena_game(
                    &config,
                    game_index,
                    opening_index,
                    player_a_is_red,
                    SmallRng::new(seed),
                    player_a_search,
                    player_b_search,
                    &player_a_name,
                    &player_b_name,
                    progress.as_ref().map(|(state, _)| Arc::clone(state)),
                )?;
                acc.push(game);
                Ok::<Vec<ArenaGameResult>, DataCollectionError>(acc)
            })
            .try_reduce(Vec::new, |mut left, mut right| {
                left.append(&mut right);
                Ok::<Vec<ArenaGameResult>, DataCollectionError>(left)
            })
    })?;

    games.sort_by_key(|game| game.game_index);

    let mut summary = ArenaSummary::new(config.clone(), player_a_name, player_b_name);
    for game in games {
        summary.record_game(game);
    }

    drop(progress);

    Ok(summary)
}

fn play_arena_game<E1, E2>(
    config: &ArenaConfig,
    game_index: u32,
    opening_index: u32,
    player_a_is_red: bool,
    mut rng: SmallRng,
    player_a_search: &Mcts<E1>,
    player_b_search: &Mcts<E2>,
    player_a_name: &str,
    player_b_name: &str,
    progress: Option<Arc<ArenaProgressState>>,
) -> Result<ArenaGameResult, DataCollectionError>
where
    E1: PositionEvaluator,
    E2: PositionEvaluator,
{
    let mut state = GameState::new();
    let mut player_a_moves = 0u64;
    let mut player_b_moves = 0u64;
    let mut player_a_search_millis = 0u64;
    let mut player_b_search_millis = 0u64;
    let mut player_a_total_nodes = 0u64;
    let mut player_b_total_nodes = 0u64;
    let mut player_a_leaf_evaluations = 0u64;
    let mut player_b_leaf_evaluations = 0u64;

    if let Some(progress) = &progress {
        progress.game_started();
    }

    loop {
        if let Some(terminal) = state.terminal_status() {
            let outcome = arena_outcome(terminal, player_a_is_red);
            if let Some(progress) = &progress {
                progress.record_game_finished(&outcome);
            }
            return Ok(ArenaGameResult {
                game_index,
                opening_index,
                player_a_seat: if player_a_is_red { ArenaSeat::Red } else { ArenaSeat::Blue },
                player_b_seat: if player_a_is_red { ArenaSeat::Blue } else { ArenaSeat::Red },
                winner: match outcome {
                    ArenaOutcome::PlayerAWin => Some(player_a_name.to_string()),
                    ArenaOutcome::PlayerBWin => Some(player_b_name.to_string()),
                    ArenaOutcome::Draw => None,
                },
                terminal: terminal_label(terminal).to_string(),
                total_plies: state.ply() as u64,
                player_a_moves,
                player_b_moves,
                player_a_search_millis,
                player_b_search_millis,
                player_a_total_nodes,
                player_b_total_nodes,
                player_a_leaf_evaluations,
                player_b_leaf_evaluations,
                outcome,
            });
        }

        let current_player_is_a = if player_a_is_red {
            state.turn() == Player::Red
        } else {
            state.turn() == Player::Blue
        };

        let started_at = Instant::now();
        let search_result = if current_player_is_a {
            player_a_search.search(&state)?
        } else {
            player_b_search.search(&state)?
        };
        let elapsed_millis = started_at.elapsed().as_millis().min(u64::MAX as u128) as u64;

        if current_player_is_a {
            player_a_moves += 1;
            player_a_search_millis += elapsed_millis;
            player_a_total_nodes += search_result.total_nodes as u64;
            player_a_leaf_evaluations += search_result.leaf_evaluations as u64;
        } else {
            player_b_moves += 1;
            player_b_search_millis += elapsed_millis;
            player_b_total_nodes += search_result.total_nodes as u64;
            player_b_leaf_evaluations += search_result.leaf_evaluations as u64;
        }

        if let Some(progress) = &progress {
            progress.record_search(
                current_player_is_a,
                elapsed_millis,
                search_result.total_nodes as u64,
                search_result.leaf_evaluations as u64,
                config.slow_search_log_millis,
            );
        }

        if search_result.action_statistics.is_empty() {
            if let Some(progress) = &progress {
                progress.record_game_finished(&ArenaOutcome::Draw);
            }
            return Ok(ArenaGameResult {
                game_index,
                opening_index,
                player_a_seat: if player_a_is_red { ArenaSeat::Red } else { ArenaSeat::Blue },
                player_b_seat: if player_a_is_red { ArenaSeat::Blue } else { ArenaSeat::Red },
                winner: None,
                terminal: terminal_label(TerminalStatus::Draw(DrawReason::Stalemate)).to_string(),
                total_plies: state.ply() as u64,
                player_a_moves,
                player_b_moves,
                player_a_search_millis,
                player_b_search_millis,
                player_a_total_nodes,
                player_b_total_nodes,
                player_a_leaf_evaluations,
                player_b_leaf_evaluations,
                outcome: ArenaOutcome::Draw,
            });
        }

        let action = select_action(config, &search_result, state.ply(), &mut rng);
        state.apply_action(action)?;
    }
}

fn arena_outcome(terminal: TerminalStatus, player_a_is_red: bool) -> ArenaOutcome {
    match terminal {
        TerminalStatus::Win { winner: Player::Red, .. } if player_a_is_red => ArenaOutcome::PlayerAWin,
        TerminalStatus::Win { winner: Player::Blue, .. } if !player_a_is_red => ArenaOutcome::PlayerAWin,
        TerminalStatus::Win { .. } => ArenaOutcome::PlayerBWin,
        TerminalStatus::Draw(_) => ArenaOutcome::Draw,
    }
}

fn select_action(
    config: &ArenaConfig,
    search_result: &SearchResult,
    ply: u16,
    rng: &mut SmallRng,
) -> Action {
    if ply < config.opening_temperature_plies {
        return sample_search_action(
            &search_result.action_statistics,
            config.opening_temperature,
            rng,
        );
    }

    best_action(&search_result.action_statistics).action.clone()
}

fn terminal_label(terminal: TerminalStatus) -> &'static str {
    match terminal {
        TerminalStatus::Win { winner: Player::Red, .. } => "red_win",
        TerminalStatus::Win { winner: Player::Blue, .. } => "blue_win",
        TerminalStatus::Draw(DrawReason::ThreefoldRepetition) => "draw_threefold",
        TerminalStatus::Draw(DrawReason::Stalemate) => "draw_stalemate",
        TerminalStatus::Draw(DrawReason::TurnLimit) => "draw_turn_limit",
    }
}

fn average_millis(total_millis: u64, moves: u64) -> f32 {
    if moves == 0 {
        0.0
    } else {
        total_millis as f32 / moves as f32
    }
}

fn nodes_per_second(total_nodes: u64, total_millis: u64) -> f32 {
    if total_millis == 0 {
        0.0
    } else {
        total_nodes as f32 / (total_millis as f32 / 1000.0)
    }
}

fn arena_progress_enabled(config: &ArenaConfig) -> bool {
    config.progress_log_every_plies > 0
        || config.slow_search_log_millis > 0
        || config.log_game_summaries
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
    use crate::eval::HeuristicEvaluator;
    use crate::mcts::MctsConfig;

    #[test]
    fn arena_summary_stays_balanced_for_two_games() {
        let search_a = Mcts::new(MctsConfig::default().with_simulations(1), HeuristicEvaluator::default());
        let search_b = Mcts::new(MctsConfig::default().with_simulations(1), HeuristicEvaluator::default());
        let summary = run_arena(
            ArenaConfig::default()
                .with_games(2)
                .with_opening_temperature_plies(0),
            "a",
            &search_a,
            "b",
            &search_b,
        )
        .unwrap();

        assert_eq!(summary.games_played, 2);
        assert_eq!(summary.player_a.red_games, 1);
        assert_eq!(summary.player_a.blue_games, 1);
        assert_eq!(summary.player_b.red_games, 1);
        assert_eq!(summary.player_b.blue_games, 1);
        assert_eq!(summary.player_a.wins + summary.player_b.wins + summary.draws, 2);
    }

    #[test]
    fn parallel_arena_matches_sequential_summary() {
        let config = ArenaConfig::default()
            .with_games(4)
            .with_opening_temperature_plies(0);
        let search_a = Mcts::new(MctsConfig::default().with_simulations(1), HeuristicEvaluator::default());
        let search_b = Mcts::new(MctsConfig::default().with_simulations(1), HeuristicEvaluator::default());

        let sequential = run_arena(config.clone(), "a", &search_a, "b", &search_b).unwrap();
        let parallel = run_parallel_arena(
            config,
            ParallelArenaConfig::default().with_threads(2),
            "a",
            &search_a,
            "b",
            &search_b,
        )
        .unwrap();

        assert_eq!(parallel.games_played, sequential.games_played);
        assert_eq!(parallel.draws, sequential.draws);
        assert_eq!(parallel.total_plies, sequential.total_plies);
        assert_eq!(parallel.player_a.wins, sequential.player_a.wins);
        assert_eq!(parallel.player_a.losses, sequential.player_a.losses);
        assert_eq!(parallel.player_a.draws, sequential.player_a.draws);
        assert_eq!(parallel.player_a.red_games, sequential.player_a.red_games);
        assert_eq!(parallel.player_a.blue_games, sequential.player_a.blue_games);
        assert_eq!(parallel.player_a.red_wins, sequential.player_a.red_wins);
        assert_eq!(parallel.player_a.blue_wins, sequential.player_a.blue_wins);
        assert_eq!(parallel.player_b.wins, sequential.player_b.wins);
        assert_eq!(parallel.player_b.losses, sequential.player_b.losses);
        assert_eq!(parallel.player_b.draws, sequential.player_b.draws);
        assert_eq!(parallel.player_b.red_games, sequential.player_b.red_games);
        assert_eq!(parallel.player_b.blue_games, sequential.player_b.blue_games);
        assert_eq!(parallel.player_b.red_wins, sequential.player_b.red_wins);
        assert_eq!(parallel.player_b.blue_wins, sequential.player_b.blue_wins);
        assert_eq!(parallel.game_results.len(), sequential.game_results.len());

        let parallel_results: Vec<_> = parallel
            .game_results
            .iter()
            .map(|game| {
                (
                    game.game_index,
                    game.opening_index,
                    game.player_a_seat.clone(),
                    game.player_b_seat.clone(),
                    game.outcome.clone(),
                    game.winner.clone(),
                    game.terminal.clone(),
                    game.total_plies,
                )
            })
            .collect();
        let sequential_results: Vec<_> = sequential
            .game_results
            .iter()
            .map(|game| {
                (
                    game.game_index,
                    game.opening_index,
                    game.player_a_seat.clone(),
                    game.player_b_seat.clone(),
                    game.outcome.clone(),
                    game.winner.clone(),
                    game.terminal.clone(),
                    game.total_plies,
                )
            })
            .collect();
        assert_eq!(parallel_results, sequential_results);
    }
}
