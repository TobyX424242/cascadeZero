use crate::engine::{DrawReason, GameState, TerminalStatus};
use crate::error::DataCollectionError;
use crate::model::{Action, Direction, Player, Position, BOARD_CELLS};

#[derive(Debug, Clone, PartialEq)]
pub struct Evaluation {
    pub value: f32,
    pub priors: Vec<(Action, f32)>,
}

pub trait PositionEvaluator {
    fn evaluate(&self, state: &GameState) -> Result<Evaluation, DataCollectionError>;
}

#[derive(Debug, Clone, PartialEq)]
pub struct HeuristicEvaluatorConfig {
    pub piece_weight: f32,
    pub mobility_weight: f32,
    pub capture_weight: f32,
    pub cascade_weight: f32,
    pub stack_weight: f32,
    pub center_weight: f32,
    pub pressure_weight: f32,
    pub vulnerability_weight: f32,
    pub support_weight: f32,
    pub profile_weight: f32,
    pub edge_weight: f32,
    pub action_value_weight: f32,
    pub terminal_bonus: f32,
    pub capture_bonus: f32,
    pub cascade_bonus: f32,
    pub merge_bonus: f32,
    pub escape_bonus: f32,
    pub support_action_bonus: f32,
    pub placement_support_bonus: f32,
    pub placement_edge_penalty: f32,
    pub disruption_bonus: f32,
    pub pressure_action_bonus: f32,
    pub policy_temperature: f32,
    pub value_scale: f32,
}

impl Default for HeuristicEvaluatorConfig {
    fn default() -> Self {
        Self {
            piece_weight: 0.38,
            mobility_weight: 0.08,
            capture_weight: 0.16,
            cascade_weight: 0.08,
            stack_weight: 0.06,
            center_weight: 0.05,
            pressure_weight: 0.12,
            vulnerability_weight: 0.18,
            support_weight: 0.09,
            profile_weight: 0.08,
            edge_weight: 0.07,
            action_value_weight: 1.35,
            terminal_bonus: 2.5,
            capture_bonus: 0.45,
            cascade_bonus: 0.18,
            merge_bonus: 0.12,
            escape_bonus: 0.18,
            support_action_bonus: 0.12,
            placement_support_bonus: 0.15,
            placement_edge_penalty: 0.16,
            disruption_bonus: 0.16,
            pressure_action_bonus: 0.10,
            policy_temperature: 0.55,
            value_scale: 8.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct HeuristicEvaluator {
    config: HeuristicEvaluatorConfig,
}

impl HeuristicEvaluator {
    pub fn new(config: HeuristicEvaluatorConfig) -> Self {
        Self { config }
    }

    pub fn value_for_player(&self, state: &GameState, player: Player) -> f32 {
        let value = self.evaluate_state_value(state);
        if state.turn() == player {
            value
        } else {
            -value
        }
    }

    fn evaluate_state_value(&self, state: &GameState) -> f32 {
        if let Some(status) = state.terminal_status() {
            return terminal_value_for_player(status, state.turn());
        }

        let current_player = state.turn();
        let opponent = current_player.opponent();
        let current_actions = state.legal_actions();
        let opponent_actions = state.legal_actions_for(opponent);

        let piece_diff = state.piece_count(current_player) as f32 - state.piece_count(opponent) as f32;
        let mobility_diff = current_actions.len() as f32 - opponent_actions.len() as f32;
        let capture_diff = action_kind_balance(&current_actions, &opponent_actions, |action| {
            matches!(action, Action::Eat { .. })
        });
        let cascade_diff = action_kind_balance(&current_actions, &opponent_actions, |action| {
            matches!(action, Action::Cascade { .. })
        });
        let stack_diff = state.max_stack_height(current_player) as f32
            - state.max_stack_height(opponent) as f32;
        let center_diff = center_mass(state, current_player) - center_mass(state, opponent);
        let pressure_diff = attack_pressure(state, current_player) - attack_pressure(state, opponent);
        let vulnerability_diff = vulnerable_mass(state, opponent) - vulnerable_mass(state, current_player);
        let support_diff = support_score(state, current_player) - support_score(state, opponent);
        let profile_diff = stack_profile(state, current_player) - stack_profile(state, opponent);
        let edge_diff = edge_exposure(state, opponent) - edge_exposure(state, current_player);

        let raw_value = piece_diff * self.config.piece_weight
            + mobility_diff * self.config.mobility_weight
            + capture_diff * self.config.capture_weight
            + cascade_diff * self.config.cascade_weight
            + stack_diff * self.config.stack_weight
            + center_diff * self.config.center_weight
            + pressure_diff * self.config.pressure_weight
            + vulnerability_diff * self.config.vulnerability_weight
            + support_diff * self.config.support_weight
            + profile_diff * self.config.profile_weight
            + edge_diff * self.config.edge_weight;

        (raw_value / self.config.value_scale).tanh().clamp(-1.0, 1.0)
    }

    fn score_action(&self, state: &GameState, action: &Action) -> Result<f32, DataCollectionError> {
        let current_player = state.turn();
        let opponent = current_player.opponent();
        let child = state.next_state(action.clone())?;
        let mut score = self.value_for_player(&child, current_player) * self.config.action_value_weight;

        let own_escape_gain = vulnerable_mass(state, current_player) - vulnerable_mass(&child, current_player);
        let opponent_risk_gain = vulnerable_mass(&child, opponent) - vulnerable_mass(state, opponent);
        let support_gain = support_score(&child, current_player) - support_score(state, current_player);
        let pressure_gain = attack_pressure(&child, current_player) - attack_pressure(state, current_player);
        score += own_escape_gain * self.config.escape_bonus;
        score += opponent_risk_gain * self.config.pressure_action_bonus;
        score += support_gain * self.config.support_action_bonus;
        score += pressure_gain * self.config.pressure_action_bonus;

        if let Some(status) = child.terminal_status() {
            score += terminal_value_for_player(status, current_player) * self.config.terminal_bonus;
        }

        score += match action {
            Action::Place { at } => {
                center_score(*at) * self.config.center_weight
                    + local_friendly_support(state, *at, current_player)
                        * self.config.placement_support_bonus
                    - edge_distance_penalty(*at) * self.config.placement_edge_penalty
            }
            Action::Move { from, direction } => {
                let merge_bonus = move_target(*from, *direction)
                    .and_then(|target| state.stack_at(target))
                    .is_some_and(|stack| stack.owner == current_player);
                center_bonus_for_action(*from, *direction)
                    + if merge_bonus { self.config.merge_bonus } else { 0.0 }
                    + local_friendly_support_gain(state, &child, current_player)
                        * self.config.support_action_bonus
            }
            Action::Eat { from, direction } => {
                let captured_height = move_target(*from, *direction)
                    .and_then(|target| state.stack_at(target))
                    .map(|stack| stack.height as f32)
                    .unwrap_or(1.0);
                self.config.capture_bonus
                    + center_bonus_for_action(*from, *direction)
                    + captured_height * 0.08
                    + local_disruption_score(state, *from, *direction, current_player)
                        * self.config.disruption_bonus
            }
            Action::Cascade { from, direction } => {
                self.config.cascade_bonus
                    + center_bonus_for_action(*from, *direction)
                    + cascade_lane_score(state, *from, *direction, current_player)
                    + local_disruption_score(state, *from, *direction, current_player)
                        * self.config.disruption_bonus
            }
        };

        Ok(score)
    }
}

impl PositionEvaluator for HeuristicEvaluator {
    fn evaluate(&self, state: &GameState) -> Result<Evaluation, DataCollectionError> {
        let value = self.evaluate_state_value(state);
        let legal_actions = state.legal_actions();

        if legal_actions.is_empty() {
            return Ok(Evaluation {
                value,
                priors: Vec::new(),
            });
        }

        let mut raw_scores = Vec::with_capacity(legal_actions.len());
        for action in &legal_actions {
            raw_scores.push(self.score_action(state, action)?);
        }

        let priors = softmax(&raw_scores, self.config.policy_temperature);
        Ok(Evaluation {
            value,
            priors: legal_actions.into_iter().zip(priors).collect(),
        })
    }
}

fn action_kind_balance<F>(current: &[Action], opponent: &[Action], predicate: F) -> f32
where
    F: Fn(&Action) -> bool,
{
    let current_count = current.iter().filter(|action| predicate(action)).count() as f32;
    let opponent_count = opponent.iter().filter(|action| predicate(action)).count() as f32;
    current_count - opponent_count
}

fn attack_pressure(state: &GameState, player: Player) -> f32 {
    let mut score = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };

        if stack.owner != player {
            continue;
        }

        for direction in Direction::ALL {
            let Some(target) = position.checked_step(direction) else {
                continue;
            };

            let Some(target_stack) = state.stack_at(target) else {
                continue;
            };

            if target_stack.owner == player {
                continue;
            }

            if stack.height >= target_stack.height {
                score += 1.0 + (stack.height - target_stack.height) as f32 * 0.2;
            }

            if target_stack.height >= stack.height {
                score -= 0.45 + (target_stack.height - stack.height) as f32 * 0.1;
            }
        }
    }

    score
}

fn vulnerable_mass(state: &GameState, player: Player) -> f32 {
    let mut total = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };

        if stack.owner != player {
            continue;
        }

        let mut attackers = 0.0;
        for direction in Direction::ALL {
            let Some(neighbor) = position.checked_step(direction) else {
                continue;
            };

            let Some(enemy_stack) = state.stack_at(neighbor) else {
                continue;
            };

            if enemy_stack.owner == player {
                continue;
            }

            if enemy_stack.height >= stack.height {
                attackers += 1.0 + (enemy_stack.height - stack.height) as f32 * 0.2;
            }
        }

        if attackers > 0.0 {
            total += stack.height as f32 * attackers;
        }
    }

    total
}

fn support_score(state: &GameState, player: Player) -> f32 {
    let mut total = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };

        if stack.owner != player {
            continue;
        }

        let mut adjacent_support = 0.0;
        for direction in Direction::ALL {
            let Some(neighbor) = position.checked_step(direction) else {
                continue;
            };

            let Some(friendly_stack) = state.stack_at(neighbor) else {
                continue;
            };

            if friendly_stack.owner == player {
                adjacent_support += 0.35 + friendly_stack.height as f32 * 0.08;
            }
        }

        total += adjacent_support + stack.height as f32 * 0.04;
    }

    total * 0.5
}

fn stack_profile(state: &GameState, player: Player) -> f32 {
    let mut total = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };

        if stack.owner != player {
            continue;
        }

        total += match stack.height {
            1 => -0.30,
            2 => 0.10,
            3 => 0.24,
            4 => 0.36,
            5 => 0.42,
            _ => 0.42 + (stack.height as f32 - 5.0) * 0.04,
        };
    }

    total
}

fn edge_exposure(state: &GameState, player: Player) -> f32 {
    let mut total = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };

        if stack.owner != player {
            continue;
        }

        total += edge_distance_penalty(position) * stack.height as f32;
    }

    total
}

fn center_mass(state: &GameState, player: Player) -> f32 {
    let mut score = 0.0;

    for index in 0..BOARD_CELLS {
        let position = Position::from_index(index).expect("board index is valid");
        let Some(stack) = state.stack_at(position) else {
            continue;
        };

        if stack.owner == player {
            score += center_score(position) * stack.height as f32;
        }
    }

    score
}

fn center_score(position: Position) -> f32 {
    let row_distance = (position.row as f32 - 3.5).abs();
    let col_distance = (position.col as f32 - 3.5).abs();
    (4.0 - 0.55 * (row_distance + col_distance)).max(0.0)
}

fn edge_distance_penalty(position: Position) -> f32 {
    let edge_distance = position
        .row
        .min(7 - position.row)
        .min(position.col.min(7 - position.col));

    match edge_distance {
        0 => 1.0,
        1 => 0.45,
        2 => 0.12,
        _ => 0.0,
    }
}

fn move_target(from: Position, direction: Direction) -> Option<Position> {
    from.checked_step(direction)
}

fn center_bonus_for_action(from: Position, direction: Direction) -> f32 {
    move_target(from, direction)
        .map(center_score)
        .unwrap_or(0.0)
        * 0.08
}

fn cascade_lane_score(state: &GameState, from: Position, direction: Direction, player: Player) -> f32 {
    let mut distance = 0.0;
    let mut current = from;

    for _ in 0..3 {
        let Some(next) = current.checked_step(direction) else {
            break;
        };

        distance += center_score(next) * 0.04;
        if let Some(stack) = state.stack_at(next) {
            if stack.owner == player.opponent() {
                distance += 0.12 + stack.height as f32 * 0.06;
                if edge_distance_penalty(next) > 0.0 {
                    distance += 0.08 * stack.height as f32;
                }
            } else {
                distance += 0.03 * stack.height as f32;
            }
        }

        current = next;
    }

    distance
}

fn local_friendly_support(state: &GameState, at: Position, player: Player) -> f32 {
    Direction::ALL
        .into_iter()
        .filter_map(|direction| at.checked_step(direction))
        .filter_map(|neighbor| state.stack_at(neighbor))
        .filter(|stack| stack.owner == player)
        .map(|stack| 0.4 + stack.height as f32 * 0.1)
        .sum()
}

fn local_friendly_support_gain(state: &GameState, child: &GameState, player: Player) -> f32 {
    support_score(child, player) - support_score(state, player)
}

fn local_disruption_score(state: &GameState, from: Position, direction: Direction, player: Player) -> f32 {
    let mut score = 0.0;
    let mut current = from;

    for step in 1..=4 {
        let Some(next) = current.checked_step(direction) else {
            break;
        };

        if let Some(stack) = state.stack_at(next) {
            if stack.owner == player.opponent() {
                score += 0.24 + stack.height as f32 * 0.07;
                if step == 1 {
                    score += 0.08;
                }
                if edge_distance_penalty(next) > 0.0 {
                    score += 0.06 * stack.height as f32;
                }
            }
        }

        current = next;
    }

    score
}

fn softmax(raw_scores: &[f32], temperature: f32) -> Vec<f32> {
    if raw_scores.is_empty() {
        return Vec::new();
    }

    let safe_temperature = temperature.max(0.05);
    let max_score = raw_scores
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);

    let weights: Vec<f32> = raw_scores
        .iter()
        .map(|score| ((score - max_score) / safe_temperature).exp())
        .collect();
    let normalization: f32 = weights.iter().sum();

    if normalization <= 0.0 {
        return vec![1.0 / raw_scores.len() as f32; raw_scores.len()];
    }

    weights
        .into_iter()
        .map(|weight| weight / normalization)
        .collect()
}

pub fn terminal_value_for_player(status: TerminalStatus, player: Player) -> f32 {
    match status {
        TerminalStatus::Win { winner, .. } if winner == player => 1.0,
        TerminalStatus::Win { .. } => -1.0,
        TerminalStatus::Draw(DrawReason::ThreefoldRepetition)
        | TerminalStatus::Draw(DrawReason::Stalemate)
        | TerminalStatus::Draw(DrawReason::TurnLimit) => 0.0,
    }
}
