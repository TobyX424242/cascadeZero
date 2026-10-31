use crate::error::DataCollectionError;
use crate::model::{
    Action, ActionStatistics, ActionTarget, GameStateSnapshot, Metadata, PolicyTargetSource,
    SearchMetadata, TrainingSample, BOARD_SIDE,
};
use crate::sink::DatasetSink;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct CollectorConfig {
    pub dataset_id: String,
    pub flush_interval: usize,
    pub keep_pruned_actions: bool,
    pub default_metadata: Metadata,
}

impl CollectorConfig {
    pub fn new(dataset_id: impl Into<String>) -> Self {
        Self {
            dataset_id: dataset_id.into(),
            flush_interval: 128,
            keep_pruned_actions: true,
            default_metadata: Metadata::new(),
        }
    }

    pub fn with_flush_interval(mut self, flush_interval: usize) -> Self {
        self.flush_interval = flush_interval.max(1);
        self
    }

    pub fn with_keep_pruned_actions(mut self, keep_pruned_actions: bool) -> Self {
        self.keep_pruned_actions = keep_pruned_actions;
        self
    }

    pub fn with_metadata<K>(mut self, key: K, value: Value) -> Self
    where
        K: Into<String>,
    {
        self.default_metadata.insert(key.into(), value);
        self
    }
}

impl Default for CollectorConfig {
    fn default() -> Self {
        Self::new("cascade-teacher")
    }
}

#[derive(Debug, Clone)]
pub struct SearchIterationRecord {
    pub root_state: GameStateSnapshot,
    pub action_statistics: Vec<ActionStatistics>,
    pub selected_action: Option<Action>,
    pub root_value: Option<f32>,
    pub outcome_target: Option<f32>,
    pub search: SearchMetadata,
    pub metadata: Metadata,
}

impl SearchIterationRecord {
    pub fn new(root_state: GameStateSnapshot, search: SearchMetadata) -> Self {
        Self {
            root_state,
            action_statistics: Vec::new(),
            selected_action: None,
            root_value: None,
            outcome_target: None,
            search,
            metadata: Metadata::new(),
        }
    }

    pub fn with_action_statistics(mut self, action_statistics: Vec<ActionStatistics>) -> Self {
        self.action_statistics = action_statistics;
        self
    }

    pub fn with_selected_action(mut self, selected_action: Action) -> Self {
        self.selected_action = Some(selected_action);
        self
    }

    pub fn with_root_value(mut self, root_value: f32) -> Self {
        self.root_value = Some(root_value);
        self
    }

    pub fn with_outcome_target(mut self, outcome_target: f32) -> Self {
        self.outcome_target = Some(outcome_target);
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

pub struct TrainingDataCollector<S: DatasetSink> {
    config: CollectorConfig,
    sink: S,
    next_sample_index: u64,
    buffered_samples: usize,
}

impl<S: DatasetSink> TrainingDataCollector<S> {
    pub fn new(config: CollectorConfig, sink: S) -> Self {
        Self {
            config,
            sink,
            next_sample_index: 0,
            buffered_samples: 0,
        }
    }

    pub fn record(
        &mut self,
        record: SearchIterationRecord,
    ) -> Result<TrainingSample, DataCollectionError> {
        let sample = self.build_sample(record)?;
        self.sink.write_sample(&sample)?;
        self.next_sample_index += 1;
        self.buffered_samples += 1;

        if self.buffered_samples >= self.config.flush_interval {
            self.flush()?;
        }

        Ok(sample)
    }

    pub fn flush(&mut self) -> Result<(), DataCollectionError> {
        self.sink.flush()?;
        self.buffered_samples = 0;
        Ok(())
    }

    pub fn into_sink(mut self) -> Result<S, DataCollectionError> {
        self.flush()?;
        Ok(self.sink)
    }

    fn build_sample(
        &self,
        mut record: SearchIterationRecord,
    ) -> Result<TrainingSample, DataCollectionError> {
        if record.root_state.board.side != BOARD_SIDE {
            return Err(DataCollectionError::InvalidBoardSide(record.root_state.board.side));
        }

        if !self.config.keep_pruned_actions {
            record
                .action_statistics
                .retain(|action_statistics| !action_statistics.was_pruned);
        }

        if record.action_statistics.is_empty() && record.selected_action.is_some() {
            return Err(DataCollectionError::SelectedActionWithoutActions);
        }

        if let Some(selected_action) = &record.selected_action {
            let selected_action_present = record
                .action_statistics
                .iter()
                .any(|action_statistics| action_statistics.action == *selected_action);

            if !selected_action_present {
                return Err(DataCollectionError::SelectedActionMissing);
            }
        }

        let root_visits = record
            .action_statistics
            .iter()
            .map(|action_statistics| action_statistics.visits)
            .sum();
        let (policy_source, policy_probabilities) = build_policy_targets(&record.action_statistics);
        let root_value = record
            .root_value
            .unwrap_or_else(|| derive_root_value(&record.action_statistics, root_visits));

        let actions = record
            .action_statistics
            .into_iter()
            .zip(policy_probabilities)
            .map(|(action_statistics, policy_probability)| ActionTarget {
                action_index: action_statistics.action.encoded_index(),
                action: action_statistics.action,
                visits: action_statistics.visits,
                policy_probability,
                value_sum: action_statistics.value_sum,
                value_mean: action_statistics.value_mean,
                prior: action_statistics.prior,
                was_pruned: action_statistics.was_pruned,
                metadata: action_statistics.metadata,
            })
            .collect();

        let mut metadata = self.config.default_metadata.clone();
        metadata.extend(record.metadata);

        Ok(TrainingSample {
            dataset_id: self.config.dataset_id.clone(),
            sample_index: self.next_sample_index,
            root_state: record.root_state,
            selected_action_index: record
                .selected_action
                .as_ref()
                .map(Action::encoded_index),
            selected_action: record.selected_action,
            policy_source,
            root_visits,
            root_value,
            outcome_target: record.outcome_target,
            actions,
            search: record.search,
            metadata,
        })
    }
}

fn build_policy_targets(
    action_statistics: &[ActionStatistics],
) -> (PolicyTargetSource, Vec<f32>) {
    if action_statistics.is_empty() {
        return (PolicyTargetSource::Terminal, Vec::new());
    }

    let total_visits: u32 = action_statistics
        .iter()
        .map(|action_statistics| action_statistics.visits)
        .sum();

    if total_visits > 0 {
        let normalization = total_visits as f32;
        let probabilities = action_statistics
            .iter()
            .map(|action_statistics| action_statistics.visits as f32 / normalization)
            .collect();
        return (PolicyTargetSource::VisitCount, probabilities);
    }

    // A zero-visit root can still be useful when we preserve either priors or a uniform fallback.
    let total_prior: f32 = action_statistics
        .iter()
        .filter_map(|action_statistics| action_statistics.prior)
        .filter(|prior| *prior > 0.0)
        .sum();

    if total_prior > 0.0 {
        let probabilities = action_statistics
            .iter()
            .map(|action_statistics| {
                action_statistics
                    .prior
                    .filter(|prior| *prior > 0.0)
                    .unwrap_or(0.0)
                    / total_prior
            })
            .collect();
        return (PolicyTargetSource::PriorProbability, probabilities);
    }

    let uniform_probability = 1.0 / action_statistics.len() as f32;
    (
        PolicyTargetSource::UniformFallback,
        vec![uniform_probability; action_statistics.len()],
    )
}

fn derive_root_value(action_statistics: &[ActionStatistics], root_visits: u32) -> f32 {
    if root_visits > 0 {
        let total_value_sum: f32 = action_statistics
            .iter()
            .map(|action_statistics| action_statistics.value_sum)
            .sum();
        return total_value_sum / root_visits as f32;
    }

    if action_statistics.is_empty() {
        return 0.0;
    }

    let total_prior: f32 = action_statistics
        .iter()
        .filter_map(|action_statistics| action_statistics.prior)
        .filter(|prior| *prior > 0.0)
        .sum();

    if total_prior > 0.0 {
        return action_statistics
            .iter()
            .map(|action_statistics| {
                let normalized_prior = action_statistics
                    .prior
                    .filter(|prior| *prior > 0.0)
                    .unwrap_or(0.0)
                    / total_prior;
                action_statistics.value_mean * normalized_prior
            })
            .sum();
    }

    let total_mean: f32 = action_statistics
        .iter()
        .map(|action_statistics| action_statistics.value_mean)
        .sum();
    total_mean / action_statistics.len() as f32
}
