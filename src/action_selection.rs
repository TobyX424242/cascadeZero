use crate::model::{Action, ActionStatistics};
use crate::rng::SmallRng;

pub(crate) fn sample_search_action(
    action_statistics: &[ActionStatistics],
    temperature: f32,
    rng: &mut SmallRng,
) -> Action {
    if action_statistics.len() == 1 {
        return action_statistics[0].action.clone();
    }

    let weights = sampling_weights(action_statistics, temperature);
    let selected_index = rng.sample_weighted_index(&weights);
    action_statistics[selected_index].action.clone()
}

pub(crate) fn best_action(action_statistics: &[ActionStatistics]) -> &ActionStatistics {
    action_statistics
        .iter()
        .max_by(|left, right| {
            left.visits
                .cmp(&right.visits)
                .then_with(|| left.value_mean.total_cmp(&right.value_mean))
                .then_with(|| right.action.encoded_index().cmp(&left.action.encoded_index()))
        })
        .expect("search results always contain at least one action")
}

fn sampling_weights(action_statistics: &[ActionStatistics], temperature: f32) -> Vec<f32> {
    let safe_temperature = temperature.max(0.05);
    let visits_total: u32 = action_statistics.iter().map(|statistics| statistics.visits).sum();

    let mut weights: Vec<f32> = if visits_total > 0 {
        let max_visits = action_statistics.iter().map(|s| s.visits).max().unwrap_or(0);
        let visit_threshold = (max_visits as f32 * 0.02).floor() as u32;

        action_statistics
            .iter()
            .map(|statistics| {
                if statistics.visits > visit_threshold {
                    statistics.visits as f32
                } else {
                    0.0
                }
            })
            .collect()
    } else {
        let prior_total: f32 = action_statistics
            .iter()
            .map(|statistics| statistics.prior.unwrap_or(0.0).max(0.0))
            .sum();

        if prior_total > 0.0 {
            action_statistics
                .iter()
                .map(|statistics| statistics.prior.unwrap_or(0.0).max(0.0))
                .collect()
        } else {
            vec![1.0; action_statistics.len()]
        }
    };

    let exponent = 1.0 / safe_temperature;
    for weight in &mut weights {
        *weight = if *weight > 0.0 {
            weight.powf(exponent)
        } else {
            0.0
        };
    }

    weights
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Action;

    #[test]
    fn best_action_prefers_high_visit_counts() {
        let action_a = Action::from_encoded_index(0).unwrap();
        let action_b = Action::from_encoded_index(1).unwrap();
        let stats = vec![
            ActionStatistics::new(action_a, 2, 0.5),
            ActionStatistics::new(action_b.clone(), 5, 0.1),
        ];

        assert_eq!(best_action(&stats).action, action_b);
    }
}
