use crate::action_selection::{best_action, sample_search_action};
use crate::collector::{SearchIterationRecord, TrainingDataCollector};
use crate::engine::{DrawReason, GameState, TerminalStatus};
use crate::eval::PositionEvaluator;
use crate::error::DataCollectionError;
use crate::mcts::{Mcts, RootDirichletConfig, SearchOptions, SearchResult};
use crate::model::{Action, Player, SearchMetadata};
use crate::progress::PeriodicReporter;
use crate::rng::{splitmix64, SmallRng};
use crate::sink::DatasetSink;
use rayon::prelude::*;
use serde::Serialize;
use serde_json::json;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const PROGRESS_REPORT_INTERVAL_MILLIS: u64 = 1000;
const ROOT_DIRICHLET_ALPHA_MIN: f32 = 0.1;
const ROOT_DIRICHLET_ALPHA_MAX: f32 = 0.8;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LeagueMemberTier {
    History,
    Elite,
}

impl LeagueMemberTier {
    fn as_str(self) -> &'static str {
        match self {
            Self::History => "history",
            Self::Elite => "elite",
        }
    }
}

#[derive(Debug, Clone)]
pub struct LeagueMember<'a, E> {
    pub name: String,
    pub tier: LeagueMemberTier,
    pub search: &'a Mcts<E>,
}

impl<'a, E> LeagueMember<'a, E> {
    pub fn new(name: impl Into<String>, tier: LeagueMemberTier, search: &'a Mcts<E>) -> Self {
        Self {
            name: name.into(),
            tier,
            search,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct LeaguePlayConfig {
    pub games: u32,
    pub sampling_temperature: f32,
    pub temperature_drop_ply: u16,
    pub root_dirichlet_alpha: f32,
    pub root_exploration_fraction: f32,
    pub progress_log_every_plies: u16,
    pub slow_search_log_millis: u64,
    pub log_game_summaries: bool,
    pub record_league_moves: bool,
    pub seed: u64,
}

impl LeaguePlayConfig {
    pub fn with_games(mut self, games: u32) -> Self {
        let mut normalized = games.max(2);
        if normalized % 2 == 1 {
            normalized += 1;
        }
        self.games = normalized;
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

    pub fn with_record_league_moves(mut self, record_league_moves: bool) -> Self {
        self.record_league_moves = record_league_moves;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Default for LeaguePlayConfig {
    fn default() -> Self {
        Self {
            games: 2,
            sampling_temperature: 1.0,
            temperature_drop_ply: 12,
            root_dirichlet_alpha: 10.0,
            root_exploration_fraction: 0.25,
            progress_log_every_plies: 0,
            slow_search_log_millis: 0,
            log_game_summaries: false,
            record_league_moves: false,
            seed: 0x1ea9_e55_ca5cade,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ParallelLeaguePlayConfig {
    pub threads: usize,
}

impl ParallelLeaguePlayConfig {
    pub fn with_threads(mut self, threads: usize) -> Self {
        self.threads = threads.max(1);
        self
    }
}

impl Default for ParallelLeaguePlayConfig {
    fn default() -> Self {
        Self {
            threads: std::thread::available_parallelism()
                .map(|parallelism| parallelism.get())
                .unwrap_or(1),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LeagueOpponentSummary {
    pub name: String,
    pub tier: LeagueMemberTier,
    pub games: u32,
    pub candidate_wins: u32,
    pub league_wins: u32,
    pub draws: u32,
    pub samples_written: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LeaguePlaySummary {
    pub games_played: u32,
    pub candidate_wins: u32,
    pub league_wins: u32,
    pub draws: u32,
    pub samples_written: u64,
    pub total_plies: u64,
    pub opponents: Vec<LeagueOpponentSummary>,
}

impl LeaguePlaySummary {
    fn new<E>(league_members: &[LeagueMember<'_, E>]) -> Self {
        Self {
            games_played: 0,
            candidate_wins: 0,
            league_wins: 0,
            draws: 0,
            samples_written: 0,
            total_plies: 0,
            opponents: league_members
                .iter()
                .map(|member| LeagueOpponentSummary {
                    name: member.name.clone(),
                    tier: member.tier,
                    games: 0,
                    candidate_wins: 0,
                    league_wins: 0,
                    draws: 0,
                    samples_written: 0,
                })
                .collect(),
        }
    }

    fn record_completed_game(&mut self, completed: &CompletedGameSummary) {
        self.games_played += 1;
        self.total_plies += completed.total_plies;
        self.samples_written += completed.samples_written;
        match completed.outcome {
            LeagueGameOutcome::CandidateWin => {
                self.candidate_wins += 1;
                self.opponents[completed.opponent_index].candidate_wins += 1;
            }
            LeagueGameOutcome::LeagueWin => {
                self.league_wins += 1;
                self.opponents[completed.opponent_index].league_wins += 1;
            }
            LeagueGameOutcome::Draw => {
                self.draws += 1;
                self.opponents[completed.opponent_index].draws += 1;
            }
        }
        self.opponents[completed.opponent_index].games += 1;
        self.opponents[completed.opponent_index].samples_written += completed.samples_written;
    }

    fn merge(&mut self, other: Self) {
        self.games_played += other.games_played;
        self.candidate_wins += other.candidate_wins;
        self.league_wins += other.league_wins;
        self.draws += other.draws;
        self.samples_written += other.samples_written;
        self.total_plies += other.total_plies;
        for (left, right) in self.opponents.iter_mut().zip(other.opponents) {
            left.games += right.games;
            left.candidate_wins += right.candidate_wins;
            left.league_wins += right.league_wins;
            left.draws += right.draws;
            left.samples_written += right.samples_written;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeagueGameOutcome {
    CandidateWin,
    LeagueWin,
    Draw,
}

#[derive(Debug, Clone)]
struct PendingLeagueRecord {
    role: LeagueRecordRole,
    record: SearchIterationRecord,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LeagueRecordRole {
    Candidate,
    League,
}

impl LeagueRecordRole {
    fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::League => "league",
        }
    }
}

#[derive(Debug, Clone)]
struct CompletedLeagueGame {
    game_index: u32,
    opening_index: u32,
    opponent_index: usize,
    opponent_name: String,
    opponent_tier: LeagueMemberTier,
    candidate_is_red: bool,
    total_plies: u64,
    terminal: TerminalStatus,
    outcome: LeagueGameOutcome,
    pending_records: Vec<PendingLeagueRecord>,
}

#[derive(Debug, Clone, Copy)]
struct CompletedGameSummary {
    opponent_index: usize,
    outcome: LeagueGameOutcome,
    samples_written: u64,
    total_plies: u64,
}

struct LeaguePlayProgressState {
    started_at: Instant,
    total_games: u32,
    active_games: AtomicU32,
    completed_games: AtomicU32,
    candidate_wins: AtomicU32,
    league_wins: AtomicU32,
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

impl LeaguePlayProgressState {
    fn maybe_create(config: &LeaguePlayConfig) -> Option<(Arc<Self>, PeriodicReporter)> {
        if !leagueplay_progress_enabled(config) {
            return None;
        }

        let state = Arc::new(Self {
            started_at: Instant::now(),
            total_games: config.games,
            active_games: AtomicU32::new(0),
            completed_games: AtomicU32::new(0),
            candidate_wins: AtomicU32::new(0),
            league_wins: AtomicU32::new(0),
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

    fn record_game_finished(&self, outcome: LeagueGameOutcome) {
        self.active_games.fetch_sub(1, Ordering::Relaxed);
        self.completed_games.fetch_add(1, Ordering::Relaxed);
        match outcome {
            LeagueGameOutcome::CandidateWin => {
                self.candidate_wins.fetch_add(1, Ordering::Relaxed);
            }
            LeagueGameOutcome::LeagueWin => {
                self.league_wins.fetch_add(1, Ordering::Relaxed);
            }
            LeagueGameOutcome::Draw => {
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
        let candidate_wins = self.candidate_wins.load(Ordering::Relaxed);
        let league_wins = self.league_wins.load(Ordering::Relaxed);
        let draws = self.draws.load(Ordering::Relaxed);
        let searched_plies = self.searched_plies.load(Ordering::Relaxed);
        let samples_written = self.samples_written.load(Ordering::Relaxed);
        let total_search_millis = self.total_search_millis.load(Ordering::Relaxed);
        let max_search_millis = self.max_search_millis.load(Ordering::Relaxed);
        let slow_searches = self.slow_searches.load(Ordering::Relaxed);
        let total_nodes = self.total_nodes.load(Ordering::Relaxed);
        let total_leaf_evaluations = self.total_leaf_evaluations.load(Ordering::Relaxed);

        Some(format!(
            "progress kind=leagueplay elapsed_ms={} games_completed={}/{} active_games={} candidate_wins={} league_wins={} draws={} searched_plies={} samples_written={} avg_search_ms={:.2} max_search_ms={} slow_searches={} total_nodes={} total_leafs={} nodes_per_second={:.2}",
            elapsed_ms,
            completed_games,
            self.total_games,
            active_games,
            candidate_wins,
            league_wins,
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

pub fn run_league_play<S, E>(
    config: LeaguePlayConfig,
    candidate_name: impl Into<String>,
    candidate_search: &Mcts<E>,
    league_members: &[LeagueMember<'_, E>],
    mut collector: TrainingDataCollector<S>,
) -> Result<(LeaguePlaySummary, TrainingDataCollector<S>), DataCollectionError>
where
    S: DatasetSink,
    E: PositionEvaluator,
{
    validate_league_members(league_members)?;

    let candidate_name = candidate_name.into();
    let mut summary = LeaguePlaySummary::new(league_members);
    let progress = LeaguePlayProgressState::maybe_create(&config);
    let mut search_index = 0u64;

    for game_index in 0..config.games {
        let opening_index = game_index / 2;
        let candidate_is_red = game_index % 2 == 0;
        let opponent_index = select_league_member_index(config.seed, opening_index, league_members.len());
        let game = play_league_game(
            &config,
            game_index,
            opening_index,
            candidate_is_red,
            SmallRng::new(splitmix64(
                config.seed
                    ^ (game_index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            )),
            &candidate_name,
            candidate_search,
            opponent_index,
            &league_members[opponent_index],
            || {
                let next = search_index;
                search_index += 1;
                next
            },
            progress.as_ref().map(|(state, _)| Arc::clone(state)),
            None,
        )?;
        let completed = write_completed_league_game(&mut collector, game)?;
        if let Some((state, _)) = &progress {
            state.record_samples_written(completed.samples_written);
        }
        summary.record_completed_game(&completed);
    }

    collector.flush()?;
    drop(progress);

    Ok((summary, collector))
}

pub fn run_parallel_league_play<S, E>(
    config: LeaguePlayConfig,
    parallel: ParallelLeaguePlayConfig,
    candidate_name: impl Into<String>,
    candidate_search: &Mcts<E>,
    league_members: &[LeagueMember<'_, E>],
    collector: TrainingDataCollector<S>,
) -> Result<(LeaguePlaySummary, TrainingDataCollector<S>), DataCollectionError>
where
    S: DatasetSink + Send,
    E: PositionEvaluator + Send + Sync,
{
    validate_league_members(league_members)?;

    if parallel.threads <= 1 || config.games <= 2 {
        return run_league_play(
            config,
            candidate_name,
            candidate_search,
            league_members,
            collector,
        );
    }

    let candidate_name = candidate_name.into();
    let shared_collector = Arc::new(Mutex::new(collector));
    let worker_collector = Arc::clone(&shared_collector);
    let progress = LeaguePlayProgressState::maybe_create(&config);
    let search_index = Arc::new(AtomicU64::new(0));
    let thread_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(parallel.threads)
        .build()?;

    let summary = thread_pool.install(|| {
        (0..config.games)
            .into_par_iter()
            .try_fold(
                || LeaguePlaySummary::new(league_members),
                |mut acc, game_index| {
                    let opening_index = game_index / 2;
                    let candidate_is_red = game_index % 2 == 0;
                    let opponent_index =
                        select_league_member_index(config.seed, opening_index, league_members.len());
                    let worker_id = rayon::current_thread_index().map(|value| value as u32);
                    let game = play_league_game(
                        &config,
                        game_index,
                        opening_index,
                        candidate_is_red,
                        SmallRng::new(splitmix64(
                            config.seed
                                ^ (game_index as u64 + 1)
                                    .wrapping_mul(0x9e37_79b9_7f4a_7c15),
                        )),
                        &candidate_name,
                        candidate_search,
                        opponent_index,
                        &league_members[opponent_index],
                        || search_index.fetch_add(1, Ordering::Relaxed),
                        progress.as_ref().map(|(state, _)| Arc::clone(state)),
                        worker_id,
                    )?;

                    let mut collector = worker_collector.lock().map_err(|_| {
                        DataCollectionError::ParallelExecution(
                            "collector mutex was poisoned during parallel leagueplay".to_string(),
                        )
                    })?;
                    let completed = write_completed_league_game(&mut *collector, game)?;
                    if let Some((state, _)) = &progress {
                        state.record_samples_written(completed.samples_written);
                    }
                    acc.record_completed_game(&completed);
                    Ok::<LeaguePlaySummary, DataCollectionError>(acc)
                },
            )
            .try_reduce(
                || LeaguePlaySummary::new(league_members),
                |mut left, right| {
                    left.merge(right);
                    Ok::<LeaguePlaySummary, DataCollectionError>(left)
                },
            )
    })?;

    drop(worker_collector);

    let mut collector = Arc::try_unwrap(shared_collector)
        .map_err(|_| {
            DataCollectionError::ParallelExecution(
                "parallel collector is still shared after leagueplay completion".to_string(),
            )
        })?
        .into_inner()
        .map_err(|_| {
            DataCollectionError::ParallelExecution(
                "collector mutex was poisoned when finalizing leagueplay".to_string(),
            )
        })?;
    collector.flush()?;
    drop(progress);

    Ok((summary, collector))
}

fn validate_league_members<E>(league_members: &[LeagueMember<'_, E>]) -> Result<(), DataCollectionError> {
    if league_members.is_empty() {
        return Err(DataCollectionError::InvalidConfiguration(
            "leagueplay requires at least one history or elite model".to_string(),
        ));
    }
    Ok(())
}

fn play_league_game<E, F>(
    config: &LeaguePlayConfig,
    game_index: u32,
    opening_index: u32,
    candidate_is_red: bool,
    mut rng: SmallRng,
    candidate_name: &str,
    candidate_search: &Mcts<E>,
    opponent_index: usize,
    opponent: &LeagueMember<'_, E>,
    mut next_search_index: F,
    progress: Option<Arc<LeaguePlayProgressState>>,
    worker_id: Option<u32>,
) -> Result<CompletedLeagueGame, DataCollectionError>
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
            let outcome = league_game_outcome(terminal, candidate_is_red);
            if let Some(progress) = &progress {
                progress.record_game_finished(outcome);
            }
            return Ok(CompletedLeagueGame {
                game_index,
                opening_index,
                opponent_index,
                opponent_name: opponent.name.clone(),
                opponent_tier: opponent.tier,
                candidate_is_red,
                total_plies: state.ply() as u64,
                terminal,
                outcome,
                pending_records,
            });
        }

        let current_player_is_candidate = current_player_is_candidate(&state, candidate_is_red);
        let legal_action_count = if config.root_dirichlet_alpha > 0.0
            && config.root_exploration_fraction > 0.0
        {
            state.legal_actions().len()
        } else {
            0
        };
        let root_dirichlet_alpha = dynamic_root_dirichlet_alpha(config, legal_action_count);
        let started_at = Instant::now();
        let search_result = if current_player_is_candidate {
            candidate_search.search_with_options(
                &state,
                search_options_for_league_play(config, root_dirichlet_alpha, &mut rng),
            )?
        } else {
            opponent
                .search
                .search_with_options(
                    &state,
                    search_options_for_league_play(config, root_dirichlet_alpha, &mut rng),
                )?
        };
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
                progress.record_game_finished(LeagueGameOutcome::Draw);
            }
            return Ok(CompletedLeagueGame {
                game_index,
                opening_index,
                opponent_index,
                opponent_name: opponent.name.clone(),
                opponent_tier: opponent.tier,
                candidate_is_red,
                total_plies: state.ply() as u64,
                terminal: TerminalStatus::Draw(DrawReason::Stalemate),
                outcome: LeagueGameOutcome::Draw,
                pending_records,
            });
        }

        let selected_action = select_action(config, &search_result, state.ply(), &mut rng);
        let role = if current_player_is_candidate {
            LeagueRecordRole::Candidate
        } else {
            LeagueRecordRole::League
        };

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
            .with_metadata("game_index", json!(game_index))
            .with_metadata("opening_index", json!(opening_index))
            .with_metadata("league_role", json!(role.as_str()))
            .with_metadata("league_candidate_name", json!(candidate_name))
            .with_metadata(
                "league_candidate_seat",
                json!(seat_label(candidate_is_red, LeagueRecordRole::Candidate)),
            )
            .with_metadata("league_opponent_name", json!(opponent.name.as_str()))
            .with_metadata("league_opponent_tier", json!(opponent.tier.as_str()))
            .with_metadata(
                "league_opponent_seat",
                json!(seat_label(candidate_is_red, LeagueRecordRole::League)),
            )
            .with_metadata("league_opponent_index", json!(opponent_index));
        if config.record_league_moves || role == LeagueRecordRole::Candidate {
            pending_records.push(PendingLeagueRecord { role, record });
        }

        state.apply_action(selected_action)?;
    }
}

fn write_completed_league_game<S: DatasetSink>(
    collector: &mut TrainingDataCollector<S>,
    completed_game: CompletedLeagueGame,
) -> Result<CompletedGameSummary, DataCollectionError> {
    let terminal_label = terminal_label(completed_game.terminal);
    let mut samples_written = 0u64;

    for pending in completed_game.pending_records {
        let outcome = terminal_value_for_turn(completed_game.terminal, pending.record.root_state.turn);
        collector.record(
            pending
                .record
                .with_outcome_target(outcome)
                .with_metadata("game_index", json!(completed_game.game_index))
                .with_metadata("opening_index", json!(completed_game.opening_index))
                .with_metadata("terminal", json!(terminal_label))
                .with_metadata("league_role", json!(pending.role.as_str()))
                .with_metadata(
                    "league_candidate_seat",
                    json!(seat_label(completed_game.candidate_is_red, LeagueRecordRole::Candidate)),
                )
                .with_metadata(
                    "league_opponent_seat",
                    json!(seat_label(completed_game.candidate_is_red, LeagueRecordRole::League)),
                )
                .with_metadata("league_opponent_name", json!(completed_game.opponent_name.as_str()))
                .with_metadata(
                    "league_opponent_tier",
                    json!(completed_game.opponent_tier.as_str()),
                ),
        )?;
        samples_written += 1;
    }

    Ok(CompletedGameSummary {
        opponent_index: completed_game.opponent_index,
        outcome: completed_game.outcome,
        samples_written,
        total_plies: completed_game.total_plies,
    })
}

fn current_player_is_candidate(state: &GameState, candidate_is_red: bool) -> bool {
    if candidate_is_red {
        state.turn() == Player::Red
    } else {
        state.turn() == Player::Blue
    }
}

fn seat_label(candidate_is_red: bool, role: LeagueRecordRole) -> &'static str {
    match (candidate_is_red, role) {
        (true, LeagueRecordRole::Candidate) | (false, LeagueRecordRole::League) => "red",
        (false, LeagueRecordRole::Candidate) | (true, LeagueRecordRole::League) => "blue",
    }
}

fn league_game_outcome(terminal: TerminalStatus, candidate_is_red: bool) -> LeagueGameOutcome {
    match terminal {
        TerminalStatus::Win {
            winner: Player::Red,
            ..
        } if candidate_is_red => LeagueGameOutcome::CandidateWin,
        TerminalStatus::Win {
            winner: Player::Blue,
            ..
        } if !candidate_is_red => LeagueGameOutcome::CandidateWin,
        TerminalStatus::Win { .. } => LeagueGameOutcome::LeagueWin,
        TerminalStatus::Draw(_) => LeagueGameOutcome::Draw,
    }
}

fn select_league_member_index(seed: u64, opening_index: u32, league_size: usize) -> usize {
    let mut rng = SmallRng::new(splitmix64(seed ^ opening_index as u64));
    (rng.next_u64() as usize) % league_size
}

fn select_action(
    config: &LeaguePlayConfig,
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
    config: &LeaguePlayConfig,
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

fn search_options_for_league_play(
    config: &LeaguePlayConfig,
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

fn terminal_value_for_turn(terminal: TerminalStatus, turn: Player) -> f32 {
    match terminal {
        TerminalStatus::Win { winner, .. } if winner == turn => 1.0,
        TerminalStatus::Win { .. } => -1.0,
        TerminalStatus::Draw(_) => -0.0,
    }
}

fn terminal_label(terminal: TerminalStatus) -> &'static str {
    match terminal {
        TerminalStatus::Win {
            winner: Player::Red,
            ..
        } => "red_win",
        TerminalStatus::Win {
            winner: Player::Blue,
            ..
        } => "blue_win",
        TerminalStatus::Draw(DrawReason::ThreefoldRepetition) => "draw_threefold",
        TerminalStatus::Draw(DrawReason::Stalemate) => "draw_stalemate",
        TerminalStatus::Draw(DrawReason::TurnLimit) => "draw_turn_limit",
    }
}

fn leagueplay_progress_enabled(config: &LeaguePlayConfig) -> bool {
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
    use crate::collector::CollectorConfig;
    use crate::eval::HeuristicEvaluator;
    use crate::mcts::MctsConfig;
    use crate::sink::JsonlDatasetSink;
    use serde_json::Value;
    use std::collections::BTreeMap;

    #[test]
    fn dynamic_root_dirichlet_alpha_scales_with_legal_action_count() {
        let config = LeaguePlayConfig::default().with_root_dirichlet_alpha(10.0);

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

    #[test]
    fn leagueplay_records_candidate_samples_by_default() {
        let candidate_search =
            Mcts::new(MctsConfig::default().with_simulations(1), HeuristicEvaluator::default());
        let opponent_search =
            Mcts::new(MctsConfig::default().with_simulations(1), HeuristicEvaluator::default());
        let opponent = LeagueMember::new("history-0", LeagueMemberTier::History, &opponent_search);
        let collector = TrainingDataCollector::new(
            CollectorConfig::new("test-league").with_flush_interval(1),
            JsonlDatasetSink::new(Vec::new()),
        );

        let (_summary, collector) = run_league_play(
            LeaguePlayConfig::default()
                .with_games(2)
                .with_temperature_drop_ply(0)
                .with_seed(7),
            "candidate",
            &candidate_search,
            &[opponent],
            collector,
        )
        .unwrap();

        let sink = collector.into_sink().unwrap().into_inner().unwrap();
        let output = String::from_utf8(sink).unwrap();
        assert!(!output.trim().is_empty());

        for line in output.lines() {
            let payload: Value = serde_json::from_str(line).unwrap();
            assert_eq!(payload["metadata"]["league_role"], "candidate");
            assert_eq!(payload["metadata"]["league_opponent_name"], "history-0");
        }
    }

    #[test]
    fn leagueplay_reuses_same_opponent_for_color_swapped_pair() {
        let candidate_search =
            Mcts::new(MctsConfig::default().with_simulations(1), HeuristicEvaluator::default());
        let opponent_a =
            Mcts::new(MctsConfig::default().with_simulations(1), HeuristicEvaluator::default());
        let opponent_b =
            Mcts::new(MctsConfig::default().with_simulations(1), HeuristicEvaluator::default());
        let league_members = [
            LeagueMember::new("history-0", LeagueMemberTier::History, &opponent_a),
            LeagueMember::new("elite-0", LeagueMemberTier::Elite, &opponent_b),
        ];
        let collector = TrainingDataCollector::new(
            CollectorConfig::new("test-league").with_flush_interval(1),
            JsonlDatasetSink::new(Vec::new()),
        );

        let (summary, collector) = run_league_play(
            LeaguePlayConfig::default()
                .with_games(2)
                .with_temperature_drop_ply(0)
                .with_seed(11),
            "candidate",
            &candidate_search,
            &league_members,
            collector,
        )
        .unwrap();

        assert_eq!(summary.games_played, 2);
        assert_eq!(summary.candidate_wins + summary.league_wins + summary.draws, 2);

        let sink = collector.into_sink().unwrap().into_inner().unwrap();
        let output = String::from_utf8(sink).unwrap();
        let mut opponents_by_game = BTreeMap::new();
        for line in output.lines() {
            let payload: Value = serde_json::from_str(line).unwrap();
            let game_index = payload["metadata"]["game_index"].as_u64().unwrap();
            opponents_by_game
                .entry(game_index)
                .or_insert_with(|| payload["metadata"]["league_opponent_name"].clone());
        }

        assert_eq!(opponents_by_game.len(), 2);
        assert_eq!(opponents_by_game.get(&0), opponents_by_game.get(&1));
    }
}
