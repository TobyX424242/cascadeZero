pub(crate) mod action_selection;
pub(crate) mod progress;
pub(crate) mod rng;

pub mod arena;
pub mod collector;
pub mod engine;
pub mod eval;
pub mod error;
pub mod leagueplay;
pub mod mcts;
pub mod model;
pub mod neural;
pub mod selfplay;
pub mod sink;

pub use arena::{
    run_arena, run_parallel_arena, ArenaConfig, ArenaGameResult, ArenaOutcome,
    ArenaPlayerSummary, ArenaSeat, ArenaSummary, ParallelArenaConfig,
};
pub use collector::{CollectorConfig, SearchIterationRecord, TrainingDataCollector};
pub use engine::{
    ActionMask, DrawReason, GameState, TerminalStatus, ACTION_MASK_WORDS,
    INITIAL_STACK_HEIGHT, PLACEMENTS_PER_PLAYER, PLAY_TURN_LIMIT, TOTAL_PLACEMENTS,
};
pub use eval::{terminal_value_for_player, Evaluation, HeuristicEvaluator, HeuristicEvaluatorConfig, PositionEvaluator};
pub use error::DataCollectionError;
pub use leagueplay::{
    run_league_play, run_parallel_league_play, LeagueMember, LeagueMemberTier,
    LeagueOpponentSummary, LeaguePlayConfig, LeaguePlaySummary, ParallelLeaguePlayConfig,
};
pub use mcts::{Mcts, MctsConfig, RootDirichletConfig, SearchOptions, SearchResult};
pub use model::{
    Action, ActionStatistics, ActionTarget, BoardState, CellState, Direction,
    GameStateSnapshot, Metadata, Phase, Player, PolicyTargetSource, Position, SearchMetadata,
    Stack, TrainingSample, BOARD_CELLS, BOARD_SIDE, CASCADE_ACTIONS, DIRECTION_COUNT,
    EAT_ACTIONS, MOVE_ACTIONS, PLACE_ACTIONS, TOTAL_ACTIONS,
};
pub use neural::{
    encode_teacher_feature_planes, encode_teacher_feature_planes_with_version, FeatureLayout,
    NeuralEvaluator, NeuralEvaluatorConfig, TeacherFeatureLayoutVersion,
    TeacherFeaturePlaneLayout, TEACHER_FEATURE_PLANES,
};
pub use selfplay::{
    run_parallel_self_play, ParallelSelfPlayConfig, SelfPlayConfig, SelfPlayRunner,
    SelfPlaySummary,
};
pub use sink::{DatasetSink, JsonlDatasetSink};
