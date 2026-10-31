use crate::error::DataCollectionError;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;

pub const BOARD_SIDE: u8 = 8;
pub const BOARD_CELLS: usize = (BOARD_SIDE as usize) * (BOARD_SIDE as usize);
pub const DIRECTION_COUNT: usize = 4;
pub const PLACE_ACTIONS: usize = BOARD_CELLS;
pub const MOVE_ACTIONS: usize = BOARD_CELLS * DIRECTION_COUNT;
pub const EAT_ACTIONS: usize = BOARD_CELLS * DIRECTION_COUNT;
pub const CASCADE_ACTIONS: usize = BOARD_CELLS * DIRECTION_COUNT;
pub const TOTAL_ACTIONS: usize = PLACE_ACTIONS + MOVE_ACTIONS + EAT_ACTIONS + CASCADE_ACTIONS;

pub type Metadata = BTreeMap<String, Value>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Player {
    Red,
    Blue,
}

impl Player {
    pub const fn opponent(self) -> Self {
        match self {
            Self::Red => Self::Blue,
            Self::Blue => Self::Red,
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::Red => 0,
            Self::Blue => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Placement,
    Play,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Position {
    pub row: u8,
    pub col: u8,
}

impl Position {
    pub fn new(row: u8, col: u8) -> Result<Self, DataCollectionError> {
        if row >= BOARD_SIDE || col >= BOARD_SIDE {
            return Err(DataCollectionError::OutOfBounds { row, col });
        }

        Ok(Self { row, col })
    }

    pub const fn index(self) -> usize {
        (self.row as usize * BOARD_SIDE as usize) + self.col as usize
    }

    pub fn from_index(index: usize) -> Result<Self, DataCollectionError> {
        if index >= BOARD_CELLS {
            return Err(DataCollectionError::InvalidActionEncoding(index as u16));
        }

        Ok(Self {
            row: (index / BOARD_SIDE as usize) as u8,
            col: (index % BOARD_SIDE as usize) as u8,
        })
    }

    pub fn checked_step(self, direction: Direction) -> Option<Self> {
        let (row_delta, col_delta) = direction.delta();
        let next_row = self.row as i16 + row_delta as i16;
        let next_col = self.col as i16 + col_delta as i16;

        if next_row < 0
            || next_row >= BOARD_SIDE as i16
            || next_col < 0
            || next_col >= BOARD_SIDE as i16
        {
            return None;
        }

        Some(Self {
            row: next_row as u8,
            col: next_col as u8,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Stack {
    pub owner: Player,
    pub height: u8,
}

impl Stack {
    pub fn new(owner: Player, height: u8) -> Result<Self, DataCollectionError> {
        if height == 0 {
            return Err(DataCollectionError::InvalidStackHeight);
        }

        Ok(Self { owner, height })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "state", content = "stack")]
pub enum CellState {
    Empty,
    Occupied(Stack),
}

impl Default for CellState {
    fn default() -> Self {
        Self::Empty
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BoardState {
    pub side: u8,
    pub cells: Vec<CellState>,
}

impl BoardState {
    pub fn empty() -> Self {
        Self {
            side: BOARD_SIDE,
            cells: vec![CellState::Empty; BOARD_CELLS],
        }
    }

    pub fn new(cells: Vec<CellState>) -> Result<Self, DataCollectionError> {
        if cells.len() != BOARD_CELLS {
            return Err(DataCollectionError::InvalidCellCount(cells.len()));
        }

        Ok(Self {
            side: BOARD_SIDE,
            cells,
        })
    }

    pub fn get(&self, position: Position) -> Result<&CellState, DataCollectionError> {
        let index = self.validate_position(position)?;
        Ok(&self.cells[index])
    }

    pub fn set(&mut self, position: Position, cell: CellState) -> Result<(), DataCollectionError> {
        let index = self.validate_position(position)?;
        self.cells[index] = cell;
        Ok(())
    }

    pub fn set_stack(
        &mut self,
        position: Position,
        owner: Player,
        height: u8,
    ) -> Result<(), DataCollectionError> {
        let stack = Stack::new(owner, height)?;
        self.set(position, CellState::Occupied(stack))
    }

    pub fn clear(&mut self, position: Position) -> Result<(), DataCollectionError> {
        self.set(position, CellState::Empty)
    }

    pub fn piece_count(&self, owner: Player) -> u16 {
        self.cells
            .iter()
            .map(|cell| match cell {
                CellState::Occupied(stack) if stack.owner == owner => stack.height as u16,
                _ => 0,
            })
            .sum()
    }

    pub fn stack_count(&self, owner: Player) -> usize {
        self.cells
            .iter()
            .filter(|cell| matches!(cell, CellState::Occupied(stack) if stack.owner == owner))
            .count()
    }

    fn validate_position(&self, position: Position) -> Result<usize, DataCollectionError> {
        if self.side != BOARD_SIDE {
            return Err(DataCollectionError::InvalidBoardSide(self.side));
        }

        if position.row >= self.side || position.col >= self.side {
            return Err(DataCollectionError::OutOfBounds {
                row: position.row,
                col: position.col,
            });
        }

        Ok(position.index())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GameStateSnapshot {
    pub phase: Phase,
    pub turn: Player,
    pub ply: u16,
    pub play_turn: u16,
    pub repetition_count: u8,
    pub board: BoardState,
    pub metadata: Metadata,
}

impl GameStateSnapshot {
    pub fn new(
        phase: Phase,
        turn: Player,
        board: BoardState,
    ) -> Result<Self, DataCollectionError> {
        if board.side != BOARD_SIDE {
            return Err(DataCollectionError::InvalidBoardSide(board.side));
        }

        Ok(Self {
            phase,
            turn,
            ply: 0,
            play_turn: 0,
            repetition_count: 0,
            board,
            metadata: Metadata::new(),
        })
    }

    pub fn with_ply(mut self, ply: u16) -> Self {
        self.ply = ply;
        self
    }

    pub fn with_play_turn(mut self, play_turn: u16) -> Self {
        self.play_turn = play_turn;
        self
    }

    pub fn with_repetition_count(mut self, repetition_count: u8) -> Self {
        self.repetition_count = repetition_count;
        self
    }

    pub fn with_metadata<K>(mut self, key: K, value: Value) -> Self
    where
        K: Into<String>,
    {
        self.metadata.insert(key.into(), value);
        self
    }

    pub fn red_piece_count(&self) -> u16 {
        self.board.piece_count(Player::Red)
    }

    pub fn blue_piece_count(&self) -> u16 {
        self.board.piece_count(Player::Blue)
    }

    pub fn total_piece_count(&self) -> u16 {
        self.red_piece_count() + self.blue_piece_count()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

impl Direction {
    pub const ALL: [Self; DIRECTION_COUNT] = [Self::Up, Self::Down, Self::Left, Self::Right];

    pub const fn index(self) -> usize {
        match self {
            Self::Up => 0,
            Self::Down => 1,
            Self::Left => 2,
            Self::Right => 3,
        }
    }

    pub fn from_index(index: usize) -> Result<Self, DataCollectionError> {
        match index {
            0 => Ok(Self::Up),
            1 => Ok(Self::Down),
            2 => Ok(Self::Left),
            3 => Ok(Self::Right),
            _ => Err(DataCollectionError::InvalidActionEncoding(index as u16)),
        }
    }

    pub const fn delta(self) -> (i8, i8) {
        match self {
            Self::Up => (-1, 0),
            Self::Down => (1, 0),
            Self::Left => (0, -1),
            Self::Right => (0, 1),
        }
    }

    pub const fn opposite(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
            Self::Left => Self::Right,
            Self::Right => Self::Left,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub enum Action {
    Place { at: Position },
    Move { from: Position, direction: Direction },
    Eat { from: Position, direction: Direction },
    Cascade { from: Position, direction: Direction },
}

impl Action {
    pub fn encoded_index(&self) -> u16 {
        match self {
            Self::Place { at } => at.index() as u16,
            Self::Move { from, direction } => {
                (PLACE_ACTIONS + from.index() * DIRECTION_COUNT + direction.index()) as u16
            }
            Self::Eat { from, direction } => {
                (PLACE_ACTIONS
                    + MOVE_ACTIONS
                    + from.index() * DIRECTION_COUNT
                    + direction.index()) as u16
            }
            Self::Cascade { from, direction } => {
                (PLACE_ACTIONS
                    + MOVE_ACTIONS
                    + EAT_ACTIONS
                    + from.index() * DIRECTION_COUNT
                    + direction.index()) as u16
            }
        }
    }

    pub fn from_encoded_index(index: u16) -> Result<Self, DataCollectionError> {
        let index = index as usize;

        if index < PLACE_ACTIONS {
            return Ok(Self::Place {
                at: Position::from_index(index)?,
            });
        }

        if index < PLACE_ACTIONS + MOVE_ACTIONS {
            let offset = index - PLACE_ACTIONS;
            return Ok(Self::Move {
                from: Position::from_index(offset / DIRECTION_COUNT)?,
                direction: Direction::from_index(offset % DIRECTION_COUNT)?,
            });
        }

        if index < PLACE_ACTIONS + MOVE_ACTIONS + EAT_ACTIONS {
            let offset = index - PLACE_ACTIONS - MOVE_ACTIONS;
            return Ok(Self::Eat {
                from: Position::from_index(offset / DIRECTION_COUNT)?,
                direction: Direction::from_index(offset % DIRECTION_COUNT)?,
            });
        }

        if index < TOTAL_ACTIONS {
            let offset = index - PLACE_ACTIONS - MOVE_ACTIONS - EAT_ACTIONS;
            return Ok(Self::Cascade {
                from: Position::from_index(offset / DIRECTION_COUNT)?,
                direction: Direction::from_index(offset % DIRECTION_COUNT)?,
            });
        }

        Err(DataCollectionError::InvalidActionEncoding(index as u16))
    }

    pub fn flipped(&self) -> Self {
        match self {
            Self::Place { at } => Self::Place {
                at: Position { row: 7 - at.row, col: 7 - at.col },
            },
            Self::Move { from, direction } => Self::Move {
                from: Position { row: 7 - from.row, col: 7 - from.col },
                direction: direction.opposite(),
            },
            Self::Eat { from, direction } => Self::Eat {
                from: Position { row: 7 - from.row, col: 7 - from.col },
                direction: direction.opposite(),
            },
            Self::Cascade { from, direction } => Self::Cascade {
                from: Position { row: 7 - from.row, col: 7 - from.col },
                direction: direction.opposite(),
            },
        }
    }

    pub fn perspective_encoded_index(&self, turn: Player) -> u16 {
        match turn {
            Player::Red => self.encoded_index(),
            Player::Blue => self.flipped().encoded_index(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionStatistics {
    pub action: Action,
    pub visits: u32,
    pub value_sum: f32,
    pub value_mean: f32,
    pub prior: Option<f32>,
    pub was_pruned: bool,
    pub metadata: Metadata,
}

impl ActionStatistics {
    pub fn new(action: Action, visits: u32, value_sum: f32) -> Self {
        let value_mean = if visits == 0 {
            0.0
        } else {
            value_sum / visits as f32
        };

        Self {
            action,
            visits,
            value_sum,
            value_mean,
            prior: None,
            was_pruned: false,
            metadata: Metadata::new(),
        }
    }

    pub fn with_prior(mut self, prior: f32) -> Self {
        self.prior = Some(prior);
        self
    }

    pub fn with_pruned(mut self, was_pruned: bool) -> Self {
        self.was_pruned = was_pruned;
        self
    }

    pub fn with_metadata<K>(mut self, key: K, value: Value) -> Self
    where
        K: Into<String>,
    {
        self.metadata.insert(key.into(), value);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ActionTarget {
    pub action: Action,
    pub action_index: u16,
    pub visits: u32,
    pub policy_probability: f32,
    pub value_sum: f32,
    pub value_mean: f32,
    pub prior: Option<f32>,
    pub was_pruned: bool,
    pub metadata: Metadata,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PolicyTargetSource {
    VisitCount,
    PriorProbability,
    UniformFallback,
    Terminal,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct SearchMetadata {
    pub iteration_index: u64,
    pub worker_id: Option<u32>,
    pub root_parallel_group: Option<u32>,
    pub total_nodes: Option<u32>,
    pub leaf_evaluations: Option<u32>,
    pub pruned_actions: u32,
    pub elapsed_millis: Option<u64>,
    pub metadata: Metadata,
}

impl SearchMetadata {
    pub fn new(iteration_index: u64) -> Self {
        Self {
            iteration_index,
            ..Self::default()
        }
    }

    pub fn with_worker_id(mut self, worker_id: u32) -> Self {
        self.worker_id = Some(worker_id);
        self
    }

    pub fn with_root_parallel_group(mut self, root_parallel_group: u32) -> Self {
        self.root_parallel_group = Some(root_parallel_group);
        self
    }

    pub fn with_total_nodes(mut self, total_nodes: u32) -> Self {
        self.total_nodes = Some(total_nodes);
        self
    }

    pub fn with_leaf_evaluations(mut self, leaf_evaluations: u32) -> Self {
        self.leaf_evaluations = Some(leaf_evaluations);
        self
    }

    pub fn with_pruned_actions(mut self, pruned_actions: u32) -> Self {
        self.pruned_actions = pruned_actions;
        self
    }

    pub fn with_elapsed_millis(mut self, elapsed_millis: u64) -> Self {
        self.elapsed_millis = Some(elapsed_millis);
        self
    }

    pub fn with_metadata<K>(mut self, key: K, value: Value) -> Self
    where
        K: Into<String>,
    {
        self.metadata.insert(key.into(), value);
        self
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TrainingSample {
    pub dataset_id: String,
    pub sample_index: u64,
    pub root_state: GameStateSnapshot,
    pub selected_action: Option<Action>,
    pub selected_action_index: Option<u16>,
    pub policy_source: PolicyTargetSource,
    pub root_visits: u32,
    pub root_value: f32,
    pub outcome_target: Option<f32>,
    pub actions: Vec<ActionTarget>,
    pub search: SearchMetadata,
    pub metadata: Metadata,
}

impl TrainingSample {
    pub fn is_terminal(&self) -> bool {
        self.actions.is_empty()
    }
}
