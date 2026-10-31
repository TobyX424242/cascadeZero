use crate::error::DataCollectionError;
use crate::model::{
    Action, BoardState, CellState, Direction, GameStateSnapshot, Phase, Player, Position, Stack,
    TOTAL_ACTIONS, BOARD_CELLS,
};
use std::cmp::Ordering;

pub const INITIAL_STACK_HEIGHT: u8 = 3;
pub const PLACEMENTS_PER_PLAYER: u8 = 4;
pub const TOTAL_PLACEMENTS: u8 = PLACEMENTS_PER_PLAYER * 2;
pub const PLAY_TURN_LIMIT: u16 = 300;
pub const ACTION_MASK_WORDS: usize = (TOTAL_ACTIONS + 63) / 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawReason {
    ThreefoldRepetition,
    Stalemate,
    TurnLimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalStatus {
    Win { winner: Player, loser: Player },
    Draw(DrawReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionMask {
    words: [u64; ACTION_MASK_WORDS],
}

impl ActionMask {
    pub fn new() -> Self {
        Self {
            words: [0; ACTION_MASK_WORDS],
        }
    }

    pub fn set(&mut self, action_index: u16) {
        let index = action_index as usize;
        self.words[index / 64] |= 1u64 << (index % 64);
    }

    pub fn contains(&self, action_index: u16) -> bool {
        let index = action_index as usize;
        (self.words[index / 64] & (1u64 << (index % 64))) != 0
    }

    pub fn count_ones(&self) -> u32 {
        self.words.iter().map(|word| word.count_ones()).sum()
    }

    pub fn words(&self) -> &[u64; ACTION_MASK_WORDS] {
        &self.words
    }

    pub fn iter_indices(&self) -> impl Iterator<Item = u16> + '_ {
        self.words.iter().enumerate().flat_map(|(word_index, word)| {
            (0..64).filter_map(move |bit_index| {
                if (word & (1u64 << bit_index)) == 0 {
                    return None;
                }

                let action_index = word_index * 64 + bit_index;
                if action_index >= TOTAL_ACTIONS {
                    None
                } else {
                    Some(action_index as u16)
                }
            })
        })
    }
}

impl Default for ActionMask {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GameState {
    phase: Phase,
    turn: Player,
    ply: u16,
    play_turn: u16,
    repetition_count: u8,
    placement_counts: [u8; 2],
    board: [Option<Stack>; BOARD_CELLS],
    play_history: Vec<u64>,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub fn new() -> Self {
        Self {
            phase: Phase::Placement,
            turn: Player::Red,
            ply: 0,
            play_turn: 0,
            repetition_count: 0,
            placement_counts: [0, 0],
            board: [None; BOARD_CELLS],
            play_history: Vec::new(),
        }
    }

    pub fn from_snapshot(snapshot: GameStateSnapshot) -> Result<Self, DataCollectionError> {
        let mut board = [None; BOARD_CELLS];
        for (index, cell) in snapshot.board.cells.into_iter().enumerate() {
            board[index] = match cell {
                CellState::Empty => None,
                CellState::Occupied(stack) => Some(stack),
            };
        }

        let placement_counts = match snapshot.phase {
            Phase::Placement => derive_placement_counts(&board)?,
            Phase::Play => [PLACEMENTS_PER_PLAYER; 2],
        };

        let mut state = Self {
            phase: snapshot.phase,
            turn: snapshot.turn,
            ply: snapshot.ply,
            play_turn: snapshot.play_turn,
            repetition_count: snapshot.repetition_count,
            placement_counts,
            board,
            play_history: Vec::new(),
        };

        if state.phase == Phase::Play {
            let stored_occurrences = state.repetition_count.max(1);
            state.repetition_count = stored_occurrences;
            state.play_history = vec![state.position_hash(); stored_occurrences as usize];
        }

        Ok(state)
    }

    pub fn to_snapshot(&self) -> GameStateSnapshot {
        let cells = self
            .board
            .iter()
            .map(|cell| match cell {
                Some(stack) => CellState::Occupied(*stack),
                None => CellState::Empty,
            })
            .collect();

        GameStateSnapshot {
            phase: self.phase,
            turn: self.turn,
            ply: self.ply,
            play_turn: self.play_turn,
            repetition_count: self.repetition_count,
            board: BoardState::new(cells).expect("engine always owns a valid 8x8 board"),
            metadata: Default::default(),
        }
    }

    pub fn phase(&self) -> Phase {
        self.phase
    }

    pub fn turn(&self) -> Player {
        self.turn
    }

    pub fn ply(&self) -> u16 {
        self.ply
    }

    pub fn play_turn(&self) -> u16 {
        self.play_turn
    }

    pub fn repetition_count(&self) -> u8 {
        self.repetition_count
    }

    pub fn placement_count(&self, player: Player) -> u8 {
        self.placement_counts[player.index()]
    }

    pub fn piece_count(&self, player: Player) -> u16 {
        self.board
            .iter()
            .flatten()
            .filter(|stack| stack.owner == player)
            .map(|stack| stack.height as u16)
            .sum()
    }

    pub fn stack_count(&self, player: Player) -> usize {
        self.board
            .iter()
            .flatten()
            .filter(|stack| stack.owner == player)
            .count()
    }

    pub fn max_stack_height(&self, player: Player) -> u8 {
        self.board
            .iter()
            .flatten()
            .filter(|stack| stack.owner == player)
            .map(|stack| stack.height)
            .max()
            .unwrap_or(0)
    }

    pub fn stack_at(&self, position: Position) -> Option<Stack> {
        self.board[position.index()]
    }

    pub fn legal_actions_for(&self, player: Player) -> Vec<Action> {
        let mut clone = self.clone();
        clone.turn = player;
        clone.legal_actions()
    }

    pub fn position_hash(&self) -> u64 {
        let mut hash = 0xcbf29ce484222325u64;
        hash = mix_hash(hash, self.phase as u8 as u64);
        hash = mix_hash(hash, self.turn as u8 as u64);
        for (index, cell) in self.board.iter().enumerate() {
            let cell_code = match cell {
                Some(stack) => {
                    let owner_code = match stack.owner {
                        Player::Red => 1u64,
                        Player::Blue => 17u64,
                    };
                    owner_code + stack.height as u64
                }
                None => 0,
            };
            hash = mix_hash(hash, (index as u64) ^ (cell_code << 8));
        }
        hash
    }

    pub fn with_stack(
        mut self,
        position: Position,
        owner: Player,
        height: u8,
    ) -> Result<Self, DataCollectionError> {
        self.set_stack(position, owner, height)?;
        Ok(self)
    }

    pub fn set_stack(
        &mut self,
        position: Position,
        owner: Player,
        height: u8,
    ) -> Result<(), DataCollectionError> {
        self.set_cell(position, Some(Stack::new(owner, height)?));
        Ok(())
    }

    pub fn clear_position(&mut self, position: Position) {
        self.set_cell(position, None);
    }

    pub fn legal_actions(&self) -> Vec<Action> {
        match self.phase {
            Phase::Placement => self.legal_placement_actions(),
            Phase::Play => self.legal_play_actions(),
        }
    }

    pub fn legal_action_mask(&self) -> ActionMask {
        let mut mask = ActionMask::new();
        for action in self.legal_actions() {
            mask.set(action.encoded_index());
        }
        mask
    }

    pub fn is_legal_action(&self, action: &Action) -> bool {
        match self.phase {
            Phase::Placement => match action {
                Action::Place { at } => self.is_legal_placement(*at),
                _ => false,
            },
            Phase::Play => self.is_legal_play_action(action),
        }
    }

    pub fn next_state(&self, action: Action) -> Result<Self, DataCollectionError> {
        let mut next_state = self.clone();
        next_state.apply_action(action)?;
        Ok(next_state)
    }

    pub fn apply_action(&mut self, action: Action) -> Result<(), DataCollectionError> {
        match self.phase {
            Phase::Placement => self.apply_placement(action)?,
            Phase::Play => self.apply_play_action(action)?,
        }

        self.ply += 1;

        match self.phase {
            Phase::Placement => {
                self.turn = self.turn.opponent();

                if self.total_placements() == TOTAL_PLACEMENTS {
                    self.phase = Phase::Play;
                    self.play_turn = 0;
                    self.repetition_count = 1;
                    self.play_history.clear();
                    self.play_history.push(self.position_hash());
                }
            }
            Phase::Play => {
                self.play_turn += 1;
                self.turn = self.turn.opponent();

                let hash = self.position_hash();
                self.play_history.push(hash);
                self.repetition_count = self
                    .play_history
                    .iter()
                    .filter(|stored_hash| **stored_hash == hash)
                    .count() as u8;
            }
        }

        Ok(())
    }

    pub fn terminal_status(&self) -> Option<TerminalStatus> {
        if self.phase != Phase::Play {
            return None;
        }

        let red_pieces = self.piece_count(Player::Red);
        let blue_pieces = self.piece_count(Player::Blue);

        if red_pieces == 0 && blue_pieces == 0 {
            return Some(TerminalStatus::Win {
                winner: self.turn,
                loser: self.turn.opponent(),
            });
        }

        if red_pieces == 0 {
            return Some(TerminalStatus::Win {
                winner: Player::Blue,
                loser: Player::Red,
            });
        }

        if blue_pieces == 0 {
            return Some(TerminalStatus::Win {
                winner: Player::Red,
                loser: Player::Blue,
            });
        }

        if self.repetition_count >= 3 {
            return Some(TerminalStatus::Draw(DrawReason::ThreefoldRepetition));
        }

        if self.legal_actions().is_empty() {
            return Some(TerminalStatus::Draw(DrawReason::Stalemate));
        }

        if self.play_turn >= PLAY_TURN_LIMIT {
            return Some(match red_pieces.cmp(&blue_pieces) {
                Ordering::Greater => TerminalStatus::Win {
                    winner: Player::Red,
                    loser: Player::Blue,
                },
                Ordering::Less => TerminalStatus::Win {
                    winner: Player::Blue,
                    loser: Player::Red,
                },
                Ordering::Equal => TerminalStatus::Draw(DrawReason::TurnLimit),
            });
        }

        None
    }

    fn total_placements(&self) -> u8 {
        self.placement_counts[Player::Red.index()] + self.placement_counts[Player::Blue.index()]
    }

    fn legal_placement_actions(&self) -> Vec<Action> {
        let mut actions = Vec::with_capacity(BOARD_CELLS);
        for index in 0..BOARD_CELLS {
            let position = Position::from_index(index).expect("board index is valid");
            if self.is_legal_placement(position) {
                actions.push(Action::Place { at: position });
            }
        }
        actions
    }

    fn legal_play_actions(&self) -> Vec<Action> {
        let mut actions = Vec::new();

        for index in 0..BOARD_CELLS {
            let from = Position::from_index(index).expect("board index is valid");
            let Some(stack) = self.stack_at(from) else {
                continue;
            };

            if stack.owner != self.turn {
                continue;
            }

            for direction in Direction::ALL {
                if let Some(target) = from.checked_step(direction) {
                    match self.stack_at(target) {
                        None => actions.push(Action::Move { from, direction }),
                        Some(target_stack) if target_stack.owner == self.turn => {
                            actions.push(Action::Move { from, direction });
                        }
                        Some(target_stack) if stack.height >= target_stack.height => {
                            actions.push(Action::Eat { from, direction });
                        }
                        _ => {}
                    }
                }

                if stack.height >= 2 {
                    actions.push(Action::Cascade { from, direction });
                }
            }
        }

        actions
    }

    fn is_legal_placement(&self, at: Position) -> bool {
        if self.phase != Phase::Placement {
            return false;
        }

        if self.placement_counts[self.turn.index()] >= PLACEMENTS_PER_PLAYER {
            return false;
        }

        if self.stack_at(at).is_some() {
            return false;
        }

        if self.total_placements() == 0 {
            return true;
        }

        !self.has_adjacent_opponent_stack(at, self.turn)
    }

    fn has_adjacent_opponent_stack(&self, at: Position, player: Player) -> bool {
        Direction::ALL.into_iter().any(|direction| {
            at.checked_step(direction)
                .and_then(|neighbor| self.stack_at(neighbor))
                .is_some_and(|stack| stack.owner == player.opponent())
        })
    }

    fn is_legal_play_action(&self, action: &Action) -> bool {
        match action {
            Action::Place { .. } => false,
            Action::Move { from, direction } => {
                let Some(stack) = self.stack_at(*from) else {
                    return false;
                };

                if stack.owner != self.turn {
                    return false;
                }

                let Some(target) = from.checked_step(*direction) else {
                    return false;
                };

                match self.stack_at(target) {
                    None => true,
                    Some(target_stack) => target_stack.owner == self.turn,
                }
            }
            Action::Eat { from, direction } => {
                let Some(stack) = self.stack_at(*from) else {
                    return false;
                };

                if stack.owner != self.turn {
                    return false;
                }

                let Some(target) = from.checked_step(*direction) else {
                    return false;
                };

                match self.stack_at(target) {
                    Some(target_stack) => {
                        target_stack.owner == self.turn.opponent()
                            && stack.height >= target_stack.height
                    }
                    None => false,
                }
            }
            Action::Cascade { from, .. } => self
                .stack_at(*from)
                .is_some_and(|stack| stack.owner == self.turn && stack.height >= 2),
        }
    }

    fn apply_placement(&mut self, action: Action) -> Result<(), DataCollectionError> {
        let Action::Place { at } = action else {
            return Err(DataCollectionError::IllegalAction(
                "placement phase only accepts place actions".to_string(),
            ));
        };

        if !self.is_legal_placement(at) {
            return Err(DataCollectionError::IllegalAction(format!(
                "cannot place at ({}, {})",
                at.row, at.col
            )));
        }

        self.set_cell(
            at,
            Some(Stack {
                owner: self.turn,
                height: INITIAL_STACK_HEIGHT,
            }),
        );
        self.placement_counts[self.turn.index()] += 1;
        Ok(())
    }

    fn apply_play_action(&mut self, action: Action) -> Result<(), DataCollectionError> {
        if !self.is_legal_play_action(&action) {
            return Err(DataCollectionError::IllegalAction(format!(
                "{:?} is not legal for {:?}",
                action, self.turn
            )));
        }

        match action {
            Action::Place { .. } => Err(DataCollectionError::IllegalAction(
                "play phase does not accept place actions".to_string(),
            )),
            Action::Move { from, direction } => self.apply_move(from, direction),
            Action::Eat { from, direction } => self.apply_eat(from, direction),
            Action::Cascade { from, direction } => self.apply_cascade(from, direction),
        }
    }

    fn apply_move(
        &mut self,
        from: Position,
        direction: Direction,
    ) -> Result<(), DataCollectionError> {
        let to = from.checked_step(direction).ok_or_else(|| {
            DataCollectionError::IllegalAction("move target is out of bounds".to_string())
        })?;
        let moving_stack = self.take_cell(from).ok_or_else(|| {
            DataCollectionError::IllegalAction("move source has no stack".to_string())
        })?;

        match self.take_cell(to) {
            None => self.set_cell(to, Some(moving_stack)),
            Some(target_stack) if target_stack.owner == moving_stack.owner => self.set_cell(
                to,
                Some(Stack {
                    owner: moving_stack.owner,
                    height: moving_stack.height + target_stack.height,
                }),
            ),
            Some(_) => unreachable!("illegal move targets are filtered before apply"),
        }

        Ok(())
    }

    fn apply_eat(
        &mut self,
        from: Position,
        direction: Direction,
    ) -> Result<(), DataCollectionError> {
        let to = from.checked_step(direction).ok_or_else(|| {
            DataCollectionError::IllegalAction("eat target is out of bounds".to_string())
        })?;
        let attacker = self.take_cell(from).ok_or_else(|| {
            DataCollectionError::IllegalAction("eat source has no stack".to_string())
        })?;

        if self.take_cell(to).is_none() {
            return Err(DataCollectionError::IllegalAction(
                "eat target has no stack".to_string(),
            ));
        }

        self.set_cell(to, Some(attacker));
        Ok(())
    }

    fn apply_cascade(
        &mut self,
        from: Position,
        direction: Direction,
    ) -> Result<(), DataCollectionError> {
        let source_stack = self.take_cell(from).ok_or_else(|| {
            DataCollectionError::IllegalAction("cascade source has no stack".to_string())
        })?;

        for distance in 1..=source_stack.height {
            if let Some(target) = step_distance(from, direction, distance) {
                self.insert_with_push(
                    target,
                    Stack {
                        owner: source_stack.owner,
                        height: 1,
                    },
                    direction,
                );
            }
        }

        Ok(())
    }

    fn insert_with_push(&mut self, position: Position, stack: Stack, direction: Direction) {
        if self.stack_at(position).is_some() {
            self.push_chain(position, direction);
        }

        self.set_cell(position, Some(stack));
    }

    fn push_chain(&mut self, position: Position, direction: Direction) {
        let Some(stack) = self.take_cell(position) else {
            return;
        };

        if let Some(next_position) = position.checked_step(direction) {
            if self.stack_at(next_position).is_some() {
                self.push_chain(next_position, direction);
            }
            self.set_cell(next_position, Some(stack));
        }
    }

    fn set_cell(&mut self, position: Position, stack: Option<Stack>) {
        self.board[position.index()] = stack;
    }

    fn take_cell(&mut self, position: Position) -> Option<Stack> {
        let index = position.index();
        let current = self.board[index];
        self.board[index] = None;
        current
    }
}

fn mix_hash(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x100000001b3)
}

fn derive_placement_counts(board: &[Option<Stack>; BOARD_CELLS]) -> Result<[u8; 2], DataCollectionError> {
    let red_pieces: u16 = board
        .iter()
        .flatten()
        .filter(|stack| stack.owner == Player::Red)
        .map(|stack| stack.height as u16)
        .sum();
    let blue_pieces: u16 = board
        .iter()
        .flatten()
        .filter(|stack| stack.owner == Player::Blue)
        .map(|stack| stack.height as u16)
        .sum();

    if red_pieces % INITIAL_STACK_HEIGHT as u16 != 0 || blue_pieces % INITIAL_STACK_HEIGHT as u16 != 0
    {
        return Err(DataCollectionError::IllegalAction(
            "placement snapshots must consist of height-3 placements only".to_string(),
        ));
    }

    let red_placements = (red_pieces / INITIAL_STACK_HEIGHT as u16) as u8;
    let blue_placements = (blue_pieces / INITIAL_STACK_HEIGHT as u16) as u8;

    if red_placements > PLACEMENTS_PER_PLAYER || blue_placements > PLACEMENTS_PER_PLAYER {
        return Err(DataCollectionError::IllegalAction(
            "placement snapshots cannot exceed four placements per player".to_string(),
        ));
    }

    Ok([red_placements, blue_placements])
}

fn step_distance(from: Position, direction: Direction, distance: u8) -> Option<Position> {
    let mut current = from;
    for _ in 0..distance {
        current = current.checked_step(direction)?;
    }
    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_phase_rejects_adjacent_opponent_cells() {
        let mut state = GameState::new();
        let opening = Position::new(3, 3).unwrap();

        state.apply_action(Action::Place { at: opening }).unwrap();

        assert!(!state.is_legal_action(&Action::Place {
            at: Position::new(2, 3).unwrap(),
        }));
        assert!(!state.is_legal_action(&Action::Place {
            at: Position::new(4, 3).unwrap(),
        }));
        assert!(!state.is_legal_action(&Action::Place {
            at: Position::new(3, 2).unwrap(),
        }));
        assert!(!state.is_legal_action(&Action::Place {
            at: Position::new(3, 4).unwrap(),
        }));
        assert!(state.is_legal_action(&Action::Place {
            at: Position::new(2, 2).unwrap(),
        }));
    }

    #[test]
    fn cascade_pushes_existing_chain_forward() {
        let mut board = BoardState::empty();
        board
            .set_stack(Position::new(0, 0).unwrap(), Player::Red, 3)
            .unwrap();
        board
            .set_stack(Position::new(0, 1).unwrap(), Player::Blue, 2)
            .unwrap();
        board
            .set_stack(Position::new(0, 2).unwrap(), Player::Blue, 1)
            .unwrap();

        let snapshot = GameStateSnapshot::new(Phase::Play, Player::Red, board).unwrap();
        let mut state = GameState::from_snapshot(snapshot).unwrap();
        state
            .apply_action(Action::Cascade {
                from: Position::new(0, 0).unwrap(),
                direction: Direction::Right,
            })
            .unwrap();

        assert_eq!(state.stack_at(Position::new(0, 0).unwrap()), None);
        assert_eq!(
            state.stack_at(Position::new(0, 1).unwrap()),
            Some(Stack {
                owner: Player::Red,
                height: 1,
            })
        );
        assert_eq!(
            state.stack_at(Position::new(0, 2).unwrap()),
            Some(Stack {
                owner: Player::Red,
                height: 1,
            })
        );
        assert_eq!(
            state.stack_at(Position::new(0, 3).unwrap()),
            Some(Stack {
                owner: Player::Red,
                height: 1,
            })
        );
        assert_eq!(
            state.stack_at(Position::new(0, 4).unwrap()),
            Some(Stack {
                owner: Player::Blue,
                height: 2,
            })
        );
        assert_eq!(
            state.stack_at(Position::new(0, 5).unwrap()),
            Some(Stack {
                owner: Player::Blue,
                height: 1,
            })
        );
    }
}
